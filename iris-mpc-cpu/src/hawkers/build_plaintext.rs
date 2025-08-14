use std::sync::Arc;

use eyre::Result;
use futures::future::Shared;
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
    stores: SharedPlaintextStores,
    irises: Vec<(IrisVersionId, IrisCode, IrisCode)>,
    params: crate::hnsw::HnswParams,
    //TODO: wrap these up in some suitable config struct
    batch_size: Option<usize>,
    batch_size_error_rate: usize,
    hnsw_M: usize,
    prf_seed: &[u8; 16],
) -> Result<()> {
    let batch_size = match batch_size {
        None => 0, //TODO: dynamic batching (use genesis::batch or not?)
        Some(batch_size) => batch_size,
    };

    let mut graphs = graphs
        .unwrap_or_else(|| [GraphMem::new(), GraphMem::new()])
        .map(|g| Arc::new(g));

    let mut stores = stores;
    let iris_pairs: [Vec<(IrisVersionId, IrisCode)>; 2] = [
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
        graphs.into_iter(),
        stores.into_iter(),
        iris_pairs.into_iter()
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
                        let query = Arc::new(iris.1.clone());
                        let version = iris.0.clone();
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

    for (side, mut graph, mut store, insert_plans) in izip!(
        STORE_IDS,
        graphs.into_iter(),
        stores.into_iter(),
        results_by_side.into_iter()
    ) {
        // Mocking these because I'm not sure what they're supposed to be
        let ids = vec![None; insert_plans.len()];
        insert::insert(
            &mut store,
            Arc::get_mut(&mut graph).unwrap(),
            &searcher,
            insert_plans
                .into_iter()
                .map(|(_, plan)| Some(plan))
                .collect(),
            &ids,
        )
        .await?;
    }

    Ok(())
}
