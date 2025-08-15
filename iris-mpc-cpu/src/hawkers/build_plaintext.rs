use std::sync::Arc;

use eyre::Result;
use iris_mpc_common::{iris_db::iris::IrisCode, IrisVersionId};
use itertools::{izip, Itertools};
use tokio::task::JoinSet;

use crate::{
    execution::hawk_main::{
        insert::{self, InsertPlanV},
        BothEyes, STORE_IDS,
    },
    genesis::BatchSize,
    hawkers::plaintext_store::SharedPlaintextStore,
    hnsw::{graph, GraphMem, HnswSearcher},
};

pub type SharedPlaintextGraphs = BothEyes<GraphMem<SharedPlaintextStore>>;
pub type SharedPlaintextStores = BothEyes<SharedPlaintextStore>;

pub async fn plaintext_batch_insert(
    graphs: Option<SharedPlaintextGraphs>,
    stores: Option<SharedPlaintextStores>,
    irises: Vec<(IrisVersionId, IrisCode, IrisCode)>,
    //TODO: wrap these up in some suitable config struct
    params: crate::hnsw::HnswParams,
    batch_size: Option<usize>,
    batch_size_error_rate: usize,
    hnsw_M: usize,
    prf_seed: &[u8; 16],
) -> Result<(SharedPlaintextGraphs, SharedPlaintextStores)> {
    let batch_size = match batch_size {
        None => 0, //TODO: dynamic batching (use genesis::batch or not?)
        Some(batch_size) => batch_size,
    };

    // Checks for same option case, but otherwise assumes graphs and stores are in sync.
    assert!(graphs.is_none() == stores.is_none());
    let graphs = graphs
        .unwrap_or_else(|| [GraphMem::new(), GraphMem::new()])
        .map(|g| Arc::new(g));
    let stores =
        stores.unwrap_or_else(|| [SharedPlaintextStore::new(), SharedPlaintextStore::new()]);

    let irises_by_side: [Vec<(IrisVersionId, IrisCode)>; 2] = [
        irises
            .iter()
            .map(|(id, left, _)| (id.clone(), left.clone()))
            .collect(),
        irises
            .iter()
            .map(|(id, _, right)| (id.clone(), right.clone()))
            .collect(),
    ];

    let mut jobs: JoinSet<Result<_>> = JoinSet::new();
    let searcher = HnswSearcher { params };

    for (side, graph, store, irises) in izip!(
        STORE_IDS,
        graphs.clone().into_iter(),
        stores.clone().into_iter(),
        irises_by_side.into_iter()
    ) {
        for batch in &irises.into_iter().enumerate().chunks(batch_size) {
            let prf_seed = prf_seed.clone();
            let mut store = store.clone();
            let graph = graph.clone();
            let searcher = searcher.clone();
            let batch = batch.collect_vec();

            jobs.spawn({
                async move {
                    let mut results = Vec::new();
                    for (index, iris) in batch {
                        let query = Arc::new(iris.1);
                        let version = iris.0;
                        let insertion_layer =
                            searcher.select_layer_prf(&prf_seed, &(version, side))?;

                        let (links, set_ep) = searcher
                            .search_to_insert(&mut store, &graph, &query, insertion_layer)
                            .await?;

                        let insert_plan: InsertPlanV<SharedPlaintextStore> = InsertPlanV {
                            query,
                            links,
                            set_ep,
                        };

                        let iside = STORE_IDS.iter().position(|&s| s == side).unwrap();
                        results.push((iside, index, insert_plan));
                    }
                    Ok(results)
                }
            });
        }
    }

    // Flatten all results, sort by side and index to recover order
    let results: Vec<_> = jobs
        .join_all()
        .await
        .into_iter()
        .collect::<Result<_, _>>()?;

    let mut results: Vec<(usize, usize, InsertPlanV<SharedPlaintextStore>)> =
        results.into_iter().flatten().collect();
    results.sort_by_key(|(iside, index, _)| (*iside, *index));

    let mut results_by_side: [Vec<_>; 2] = [Vec::new(), Vec::new()];
    for (iside, index, insert_plan) in results {
        results_by_side[iside].push((index, insert_plan));
    }

    let mut ret_graphs = Vec::new();
    let mut ret_stores = Vec::new();

    for (_side, graph, mut store, insert_plans) in izip!(
        STORE_IDS,
        graphs.into_iter(),
        stores.into_iter(),
        results_by_side.into_iter()
    ) {
        // Mocking these because I'm not sure what they're supposed to be
        let ids = vec![None; insert_plans.len()];
        // Should be able to take ownership from Arc, as all threads have finished before.
        let mut graph = Arc::try_unwrap(graph).unwrap();
        insert::insert(
            &mut store,
            &mut graph,
            &searcher,
            insert_plans
                .into_iter()
                .map(|(_, plan)| Some(plan))
                .collect(),
            &ids,
        )
        .await?;

        ret_graphs.push(graph);
        ret_stores.push(store);
    }

    Ok((
        ret_graphs.try_into().unwrap(),
        ret_stores.try_into().unwrap(),
    ))
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        hawkers::plaintext_store::PlaintextStore,
        hnsw::{graph::layered_graph::migrate, HnswParams, HnswSearcher},
    };
    use aes_prng::AesRng;
    use iris_mpc_common::iris_db::db::IrisDB;
    use rand::SeedableRng;

    #[tokio::test]
    async fn test_something() -> Result<()> {
        let mut rng = AesRng::seed_from_u64(0_u64);
        let database_size = 512;
        let searcher = HnswSearcher::new_with_test_parameters();

        let mut graphs = Vec::new();
        let mut stores = Vec::new();
        let mut irises = Vec::new();

        for side in STORE_IDS {
            let mut ptxt_vector = PlaintextStore::new_random(&mut rng, database_size);
            let ptxt_graph = ptxt_vector
                .generate_graph(&mut rng, database_size, &searcher)
                .await?;

            let mut shared_vector = SharedPlaintextStore::from(ptxt_vector);
            let graph = migrate(ptxt_graph, |id| id);
            let irises_ = IrisDB::new_random_rng(1024, &mut rng).db;
            graphs.push(graph);
            stores.push(shared_vector);
            irises.push(irises_);
        }

        let irises = izip!(irises[0].clone(), irises[1].clone())
            .enumerate()
            .map(|(id, (left, right))| (0, left, right))
            .collect();

        let prf_seed = [0u8; 16];

        let (graphs, stores) = plaintext_batch_insert(
            Some(graphs.try_into().unwrap()),
            Some(stores.try_into().unwrap()),
            irises,
            HnswParams::new(64, 32, 32),
            Some(256),
            0,
            0,
            &prf_seed,
        )
        .await?;

        assert!(stores[0].storage.data.read().await.points.len() == 512 + 1024);
        assert!(stores[1].storage.data.read().await.points.len() == 512 + 1024);
        Ok(())
    }
}
