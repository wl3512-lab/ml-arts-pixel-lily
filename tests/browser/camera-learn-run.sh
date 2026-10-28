#!/bin/zsh
# Does the CNN actually learn through the real camera path (mirror, right-half
# crop, 64x64, centered pixels)? Chrome's fake camera plays the clip from
# make-shapes-camera.py: big disc / small disc / nothing on the stem-hand side,
# with a corner marker saying which phase is showing. Examples are labeled from
# the marker, the model trains, then its live guesses are compared to the marker.
# Usage: camera-learn-run.sh <sketch url> <clip.y4m>   (needs agent-browser)
export PATH="$HOME/.local/bin:$PATH" AGENT_BROWSER_SESSION=${AGENT_BROWSER_SESSION:-a8learn}
export AGENT_BROWSER_ARGS="--use-fake-ui-for-media-stream,--use-fake-device-for-media-stream,--use-file-for-fake-video-capture=$2"
URL="$1"
t(){ perl -e 'alarm 120; exec @ARGV' agent-browser "$@"; }
q(){ t eval "$1" | tr -d '\\' | sed 's/^"//;s/"$//'; }
PHASE="(() => { const v = [0, 1, 2].map(p => video.get(24 + p * 50, 25)[0]); const m = Math.max(...v); return m > 200 ? v.indexOf(m) : -1; })()"

t set viewport 1280 820 >/dev/null; t open "$URL" >/dev/null; sleep 3
t click "#startCamera" >/dev/null
for i in $(seq 1 40); do s=$(q "document.getElementById('inputStatus').textContent"); [[ "$s" == Camera\ on* || "$s" == *error* ]] && break; sleep 1; done
echo "camera: $s"
q "window.__c = setInterval(() => { const p = $PHASE; if (p >= 0 && counts()[LABELS[p]] < 40) addExample(LABELS[p]); }, 110)" >/dev/null
for i in $(seq 1 40); do c=$(q "JSON.stringify(lilyState().counts)"); [[ "$c" == '{"open":40,"fist":40,"none":40}' ]] && break; sleep 1; done
q "clearInterval(window.__c)" >/dev/null
echo "counts: $c"
# the discs only appear on the stem-hand half, so the crops themselves must get
# brighter with the disc size; a crop of the wrong half would only see noise
echo "mean crop brightness by label (centered): $(q "JSON.stringify(Object.fromEntries(LABELS.map(l => { const ex = examples.filter(e => e.label === l); return [l, +(ex.reduce((s, e) => s + e.inputs.reduce((a, v) => a + v, 0) / e.inputs.length, 0) / ex.length).toFixed(3)]; })))")"
t press t >/dev/null
for i in $(seq 1 150); do d=$(q "lilyState().trained"); [[ $d == true ]] && break; sleep 1; done
echo "train: $(q "document.getElementById('trainStatus').textContent" | tr '\n' ' ')"
echo "loss every 8 epochs: $(q "JSON.stringify(lilyState().lossHistory.filter((v,i)=>i%8==0||i==63).map(v=>+v.toFixed(3)))")"
match=0; total=0
for i in $(seq 1 30); do
  r=$(q "(() => { const p = $PHASE; return p < 0 ? 'skip' : (LABELS[p] + ' ' + lilyState().stemGuess.label); })()")
  if [[ "$r" != skip ]]; then total=$((total + 1)); [[ "${r%% *}" == "${r##* }" ]] && match=$((match + 1)); fi
  sleep 0.45
done
echo "live guesses matching the phase: $match/$total (guesses lag the clip by ~0.1-0.3 s, so a few misses at phase changes are expected)"
t errors
