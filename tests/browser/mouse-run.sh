#!/bin/zsh
# Mouse-mode check, no camera: drives headless Chrome with real mouse and key
# events through every plant state and saves a screenshot of each.
# Usage: mouse-run.sh <sketch url> [screenshot dir]   (needs agent-browser)
export PATH="$HOME/.local/bin:$PATH" AGENT_BROWSER_SESSION=${AGENT_BROWSER_SESSION:-a8mouse}
URL="$1"; M="${2:-}"
t(){ perl -e 'alarm 60; exec @ARGV' agent-browser "$@"; }
st(){ t eval "JSON.stringify((({plant:p,input,falling,fallen})=>({h:Math.round(p.height),wilt:+p.wilt.toFixed(2),bloom:+p.bloom.toFixed(2),bud:+p.bud.toFixed(2),plucks:p.plucks,pinch:input.pinch===null?null:+input.pinch.toFixed(2),stem:input.stem,falling,fallen}))(lilyState()))" | tr -d '\\' | sed 's/^"//;s/"$//'; }
shot(){ [[ -n "$M" ]] && t screenshot "$M/$1" >/dev/null; }
# garden position on screen, from the sketch itself
gx(){ t eval "Math.round(gardenRect.x + gardenRect.s * 120 * $1)" | tr -d '"'; }
gy(){ t eval "Math.round(gardenRect.y + gardenRect.s * 150 * $1)" | tr -d '"'; }

t set viewport 1280 820 >/dev/null; t open "$URL" >/dev/null; sleep 3
t click "#useMouse" >/dev/null; sleep 0.5
echo "start:          $(st)"; shot 01-start.png

t keydown o >/dev/null; sleep 4; t keyup o >/dev/null
echo "held O 4s:      $(st)"; shot 02-grown.png

Y=$(gy 0.5)
t mouse move $(gx 0.1) $Y >/dev/null; t mouse down >/dev/null
for f in 0.2 0.35 0.5 0.65 0.8 0.95; do t mouse move $(gx $f) $Y >/dev/null; sleep 0.25; done
sleep 0.6
echo "dragged right:  $(st)"; shot 03-bloom.png

# slow close: halfway back over ~2.5 s, should close without plucking
for f in 0.9 0.8 0.7 0.6 0.5 0.4; do t mouse move $(gx $f) $Y >/dev/null; sleep 0.4; done
echo "slow to 0.4:    $(st)"; shot 04-half.png
for f in 0.6 0.8 0.95; do t mouse move $(gx $f) $Y >/dev/null; sleep 0.2; done; sleep 0.6

# snap: jump straight to the left edge
t mouse move $(gx 0.02) $Y >/dev/null; sleep 0.15
echo "snapped shut:   $(st)"; sleep 0.5; shot 05-pluck.png
sleep 2.5; echo "2.5 s later:    $(st)"; shot 06-fallen.png
t mouse up >/dev/null; sleep 1.5
echo "regrown:        $(st)"

t keydown f >/dev/null; sleep 2.2; t keyup f >/dev/null
echo "held F 2.2s:    $(st)"; shot 07-wilted.png
t keydown o >/dev/null; sleep 1.5; t keyup o >/dev/null
echo "held O 1.5s:    $(st)"
t mouse move $(gx 0.1) $Y >/dev/null; t mouse down >/dev/null
for f in 0.3 0.5 0.7 0.85; do t mouse move $(gx $f) $Y >/dev/null; sleep 0.25; done; sleep 0.8
echo "bloom again:    $(st)"; shot 08-revived.png
t mouse up >/dev/null
t errors
