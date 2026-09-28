// Single API client for all backend endpoints.
async function j(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    let detail = `HTTP ${r.status}`;
    try { detail = (await r.json()).detail ?? detail; } catch { /* not json */ }
    throw new Error(`${path}: ${detail}`);
  }
  return r.json();
}

const post = (path, body) => j(path, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: body === undefined ? undefined : JSON.stringify(body),
});

export const api = {
  // review
  queue: (sort = "score") => j(`/api/queue/?sort=${sort}`),
  clusters: () => j(`/api/cluster/`),
  photo: (hash) => j(`/api/photo/${hash}`),
  decide: (body) => post(`/api/decide/`, body),
  decideAll: (body) => post(`/api/decide/all`, body),
  decideCluster: (clusterId, mode) => post(`/api/decide/cluster`, { cluster_id: clusterId, mode }),
  pending: () => j(`/api/decide/pending`),
  // pipeline
  pipelineStatus: () => j(`/api/pipeline/status`),
  runStage: (name) => post(`/api/pipeline/run/${name}`),
  autoRun: (leg) => post(`/api/pipeline/auto/${leg}`),
  cancelJob: () => post(`/api/pipeline/cancel`),
  logs: (after = 0) => j(`/api/pipeline/logs?after=${after}`),
  stageLog: (stage) => j(`/api/pipeline/stage-log/${stage}`),
  resetSession: () => post(`/api/pipeline/reset`, { confirm: "RESET" }),
  // system
  systemStats: () => j(`/api/system/stats`),
};
