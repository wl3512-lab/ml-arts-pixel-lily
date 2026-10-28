#!/bin/zsh
# Camera-path check without a person: Chrome's fake camera (a test pattern, no
# hands). Proves the plumbing: handPose loads, holding 1/2/3 streams 64x64
# crops, training runs, and the CNN keeps classifying afterwards. It can't show
# that open/fist/none are told apart; that needs a real hand.
# Usage: camera-run.sh <sketch url> [screenshot dir]   (needs agent-browser)
export PATH="$HOME/.local/bin:$PATH" AGENT_BROWSER_SESSION=${AGENT_BROWSER_SESSION:-a8cam}
export AGENT_BROWSER_ARGS="--use-fake-ui-for-media-stream,--use-fake-device-for-media-stream"
URL="$1"; M="${2:-}"
t(){ perl -e 'alarm 120; exec @ARGV' agent-browser "$@"; }
q(){ t eval "$1" | tr -d '\\' | sed 's/^"//;s/"$//'; }

t set viewport 1280 820 >/dev/null; t open "$URL" >/dev/null; sleep 3
t click "#startCamera" >/dev/null
for i in $(seq 1 40); do s=$(q "document.getElementById('inputStatus').textContent"); [[ "$s" == Camera\ on* || "$s" == *error* ]] && break; sleep 1; done
echo "camera: $s"
for k in 1 2 3; do t keydown $k >/dev/null; sleep 2.2; t keyup $k >/dev/null; done
echo "counts: $(q "JSON.stringify(lilyState().counts)")"
[[ -n "$M" ]] && t screenshot "$M/20-camera-recorded.png" >/dev/null
t press t >/dev/null
for i in $(seq 1 120); do d=$(q "lilyState().trained"); [[ $d == true ]] && break; sleep 1; done
echo "train: $(q "document.getElementById('trainStatus').textContent")"
echo "loss: $(q "JSON.stringify(lilyState().lossHistory.filter((v,i)=>i%8==0||i==63).map(v=>+v.toFixed(3)))")"
g1=$(q "JSON.stringify(lilyState().stemGuess)"); sleep 1.5; g2=$(q "JSON.stringify(lilyState().stemGuess)")
echo "guesses keep coming: $g1 -> $g2"
[[ -n "$M" ]] && t screenshot "$M/21-camera-trained.png" >/dev/null
t errors
