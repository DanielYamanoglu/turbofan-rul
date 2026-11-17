#!/usr/bin/env bash
# Build the image, run it on a throwaway kind cluster and call the service through its Kubernetes Service.
# Needs docker, kubectl and kind (KIND overrides the kind command, e.g. KIND="go run sigs.k8s.io/kind@v0.27.0").
set -euo pipefail

CLUSTER=turbofan-smoke
KIND=${KIND:-kind}
PORT=18080
cd "$(dirname "$0")/.."

cleanup() {
  [[ -n "${FORWARD_PID:-}" ]] && kill "$FORWARD_PID" 2>/dev/null || true
  [[ "${KEEP_CLUSTER:-0}" == 1 ]] || $KIND delete cluster --name "$CLUSTER" >/dev/null 2>&1 || true
}
trap cleanup EXIT

docker build -q -t turbofan-rul:local .
$KIND get clusters 2>/dev/null | grep -qx "$CLUSTER" || $KIND create cluster --name "$CLUSTER" --wait 120s
$KIND load docker-image turbofan-rul:local --name "$CLUSTER"

kubectl --context "kind-$CLUSTER" apply -k k8s
kubectl --context "kind-$CLUSTER" rollout status deployment/turbofan-rul --timeout=180s
kubectl --context "kind-$CLUSTER" get pods -l app.kubernetes.io/name=turbofan-rul -o wide

kubectl --context "kind-$CLUSTER" port-forward service/turbofan-rul "$PORT:80" >/dev/null 2>&1 &
FORWARD_PID=$!
for _ in $(seq 30); do curl -sf "localhost:$PORT/health" >/dev/null && break; sleep 1; done

echo "health: $(curl -sf "localhost:$PORT/health")"
echo "ready:  $(curl -sf "localhost:$PORT/ready")"
python3 - "$PORT" <<'EOF'
import json, sys, urllib.request
names = ["cycle"] + [f"op{i}" for i in range(1, 4)] + [f"s{i}" for i in range(1, 22)]
reading = {name: 1.0 for name in names} | {"cycle": 1}
request = urllib.request.Request(
    f"http://localhost:{sys.argv[1]}/predict",
    data=json.dumps({"history": [reading]}).encode(),
    headers={"content-type": "application/json"},
)
body = json.load(urllib.request.urlopen(request, timeout=10))
assert set(body) == {"rul_cycles", "maintenance_alert", "cycles_seen"}, body
print("predict:", body)
EOF
echo "smoke test passed"
