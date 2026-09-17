# rift_oracle

Win-probability model for League of Legends ranked games. Point it at the game
you are playing and it tells you your odds, watches them move, and tells you
what moved them — the Baron, the item that came online, the composition that
was always going to out-scale you.

```
  YOU  63.2%  ################################............  36.8% THEM
  Trending gently your way (+4.2 pts in 2 min)
```

Three things make it more than a number:

- **It explains itself exactly.** The model is additive, so the change in your
  odds decomposes into per-factor contributions that sum to the change with no
  residual. "You dropped 18 points" always comes with the arithmetic.
- **It knows what it cannot see.** The in-client API reports your gold but not
  your team's; a match timeline reports everything but only after the game. The
  model is trained under both regimes and masked features contribute exactly
  zero rather than a guess.
- **Its advice is the same model, run backwards.** Every suggestion is a real
  counterfactual: take this dragon, re-ask the model, report the difference.

---

## Quick start

```bash
git clone https://github.com/priyeshj12/LeagueWinProb
cd LeagueWinProb
pip install -e .

rift-oracle demo            # full pipeline on a simulated game, no key needed
rift-oracle doctor          # check key, network, client, model
```

Set your Riot API key once (keys come from https://developer.riotgames.com/):

```bash
export RIOT_API_KEY=RGAPI-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
# or: rift-oracle configure --api-key RGAPI-... --platform euw1
```

`rift-oracle doctor --riot-id "You#TAG"` checks the whole chain end to end:
the key, both routing values, your ranked history, timeline access, and
spectator.

Then:

```bash
rift-oracle live                        # the game you are in right now
rift-oracle watch "Faker#KR1"           # a game in progress, by Riot ID
rift-oracle replay NA1_5123456789       # a finished game, explained
rift-oracle replay NA1_5123456789 --html report.html
```

---

## The three modes, and what each can actually see

Riot exposes live game state in exactly one place: the **Live Client Data API**
on `127.0.0.1:2999`, served by your own client, for your own game, with no API
key. The Web API's spectator endpoint gives you the draft and the clock for a
game in progress but no gold and no kills. This tool does not pretend
otherwise.

| | `live` | `watch` | `replay` |
|---|---|---|---|
| Source | Live Client Data API | spectator-v5 | match-v5 + timeline |
| Needs a key | no | yes | yes |
| Whose game | yours | anyone's | anyone's, finished |
| Gold / kills / items | yes (gold estimated) | no | exact |
| Experience, damage dealt | no | no | exact |
| Who is dead right now | exact | no | derived |
| Live swing feed | yes | odds only | full post-game |

**`live`** is the one you want mid-game. It polls every couple of seconds and
draws a dashboard: current odds, the trace so far, the scoreboard, what the
model is reacting to, ranked actions, and the swings that already happened.

**`watch`** tracks someone else's game in progress. It reports the odds the
draft and the clock imply, then pulls the full timeline the moment the game
ends and explains every swing. Add `--ranks` to fold each player's ranked tier
into a pre-game prior.

Because composition scaling is a known function of the clock, `watch` also
projects the draft forward before a single minion dies. Kayle/Vayne/Veigar/
Nasus/Kassadin against Lee Sin/Pantheon/Renekton/Draven/Elise starts near 32%
and ends near 68%, crossing even right around the 22-minute mark — so the
readout is not just a number but a plan: play for tempo, or play for time.

**`replay`** is the post-mortem, and the mode with the most to say, because a
timeline has everything.

### Gold in live mode is estimated

The in-client API reports items, levels, CS and KDA for all ten players but
current gold only for you. Team gold is therefore reconstructed from two
independent signals — the summed `price` of every item held (exact for gold
already spent) and a closed-form income model (starting gold, passive income,
CS, takedowns, objectives) — with the difference used to estimate unspent
wallets. The dashboard prints how well the two agree so you can discount the
number when they disagree. `replay` has no such problem; its gold is Riot's.

---

## What the output looks like

```
18:30-20:00   [spike]
Red killed Baron Nashor  (69% -> 39%, -30.0 pts Red)
    Baron buff              -11.1 pts   Red killed Baron Nashor
    gold lead                -9.9 pts   Red killed Baron; Red took the Bot Inner turret
    turret lead              -6.8 pts   Red took the Bot Inner turret
    the fight                +5.5 pts   2 kills for Red; 1 kill for Blue
    also in window                      Smolder completed Zhonya's Hourglass (3250g)
```

and, for the current moment:

```
+   Win the next teamfight 3-for-0      +18.7 pts   Three dead is enough time to take anything.
+   Take Baron Nashor                   +12.3 pts   The buff converts directly into structures.
+   Catch out Smolder                    +5.4 pts   Their biggest wallet (9,177g, 2/7/2).
!   They take Baron                      -8.2 pts   Ward it by 19:00, do not face-check the pit.
*   Their damage is 71% physical - armor is the efficient buy (Plated Steelcaps, Thornmail).
*   Buy grievous wounds - Aatrox out-heals your burst without it.

They out-scale you (Smolder, Zeri). Their curve takes over around 22:00 -
you have about 4 minutes of tempo left.
```

`--html report.html` writes a standalone page with the same content: an
interactive trace, the factor breakdown, and every swing as a card. No network,
no build step, dark mode included.

---

## How the model works

A generalised additive model over 21 team-difference features, fit by penalised
IRLS. Two structural choices do most of the work.

**Antisymmetry.** Every feature is a blue-minus-red difference and every basis
function is odd, so negating the input negates the logit and
`P(blue) + P(red) = 1` holds to floating point. A single fitted `side_bias`
scalar carries the real blue-side edge and is the only thing in the model that
can prefer a side. There is no way for it to learn a lopsided rule it cannot
name.

**Exact attribution.** Because the logit is a plain sum of per-feature terms,
the change between two moments is the sum of the changes in those terms. No
sampling, no surrogate model, no residual. A boosted ensemble would score
slightly better and would need a second approximation layer to say anything
about causes; this is the trade the tool is built around.

Per feature the basis is `x`, two odd hinges at data-driven knots, and `x*tau`
where `tau` is a normalised clock — which is how one model knows that 3k gold
at ten minutes and 3k gold at thirty minutes are different facts.

**Monotonicity is enforced.** Every feature is defined so more is better for
blue, so each fitted response must be non-decreasing. Without the constraint,
features that correlate with each other pick up negative coefficients
conditional on their partners, and the model reports that getting kills lowered
your odds — defensible statistics, useless advice. Each Newton step is projected
onto the monotone cone (Dykstra), which also acts as a regulariser: on held-out
games the constrained fit beats the unconstrained one on log loss, accuracy,
AUC and calibration alike.

### Features

| group | features |
|---|---|
| economy | relative and absolute gold lead, experience, levels, CS, completed item value |
| combat | kills, champions alive, respawn timers, damage share |
| objectives | turrets (weighted by tier), plates, open inhibitors, dragons, soul, Elder, Baron, Herald |
| draft & macro | composition scaling, vision, ranked-tier prior |

Correlated features are summed into one line for display (the two gold
encodings become "gold lead"), so the model keeps the resolution it needs while
the report stays readable.

### Champion scaling

Each champion carries a hand-rated coefficient in `[-1, +1]` for how its
*relative* power moves across a game — Kayle at `+1.00`, Pantheon at `-0.55` —
and a team's curve is the mean, weighted by a ramp that crosses zero at 22
minutes. Champions released after this build fall back to the mean of their
Riot role tags. This is the feature that produces "their composition overtakes
yours around 28:00", and it is why the tool can tell you that you are ahead and
losing anyway.

---

## Training

The bundled model is fit on 6,000 simulated Summoner's Rift games, so the tool
works with no key, no network and no data. The simulator is a real generative
model — persistent skill gaps, composition scaling, snowballing, bounties as
negative feedback, and a siege hazard so games end when someone has pressure
rather than at a fixed clock. Games are played to the nexus falling, which
means the label comes from the same process as the features and the resulting
probabilities are calibrated by construction.

Held out on 1,200 unseen simulated games:

| metric | value |
|---|---|
| log loss | 0.506 |
| Brier | 0.170 |
| AUC | 0.826 |
| accuracy | 73.9% |
| expected calibration error | 1.2% |

Accuracy by game clock runs 64% in the first ten minutes to 85% after thirty,
which is the shape a real win-probability model has: early states genuinely do
not determine outcomes.

The ridge penalty is cross-validated by default (`--l2 auto`), because the
right amount of shrinkage depends on how much data there is — a few hundred
real matches against eighty-five parameters need far more than several thousand
simulated games do. Above a few hundred games the search runs on a subsample
and scales the result by the size ratio, which keeps the effective prior fixed
rather than the penalty.

To fit on real matches instead:

```bash
rift-oracle harvest --ladder --count 800      # seeds across the whole ladder
rift-oracle train --data matches
rift-oracle backtest --data matches --live-mask
```

`--ladder` seeds from `league-v4` across Bronze through Master, a spread chosen
because a model trained only on Challenger games would be asked about games
that look nothing like them. `--tier EMERALD:II` picks specific rungs and
`--riot-id` seeds from named accounts; they combine. Matches already on disk
are skipped, so an interrupted harvest resumes.

Throughput is capped by Riot's limits, not by the tool: at two requests per
match against a 100-per-2-minutes budget, expect roughly 25 matches a minute.

`train` splits by *game*, never by frame — consecutive states from one match
share a winner and most of their features, and a random row split would report
flattering nonsense. Every game enters training twice, once with everything
observed and once masked down to what `live` can see, so the model is honest
under both.

`backtest` reports a reliability table and how much an isotonic recalibration
would buy. On a well-fit model the answer is approximately nothing, which is
the point.

---

## Building `rift_oracle.exe`

```powershell
pip install -e .[dev]
pyinstaller packaging/rift_oracle.spec      # -> dist/rift_oracle.exe
```

`packaging/build_exe.ps1` wraps that on Windows and `packaging/build_exe.sh` on
Linux/macOS. The trained model is bundled into the executable, so the `.exe` is
self-contained: drop it next to a `.riot_api_key` file and it works.

CI builds it on every tag — see `.github/workflows/build-exe.yml`.

---

## Command reference

| command | what it does |
|---|---|
| `live` | attach to the game running on this machine |
| `watch RIOT_ID` | track a game in progress, then explain it when it ends |
| `replay MATCH_ID` | replay a finished match and explain every swing |
| `demo` | the whole pipeline on a simulated game (`--replay-speed 8` animates it) |
| `train` | fit the model, simulated or `--data` real matches |
| `harvest` | download match + timeline pairs |
| `backtest` | score the model and check its calibration |
| `doctor` | check key, network, client and model |
| `configure` | save the API key and default platform |
| `clear-cache` | drop cached Riot responses |

Useful flags: `--threshold 0.03` to catch smaller swings, `--window 120` to let
a swing span longer, `--html` and `--json-out` to write reports, `--compact`
for one-line output, `--platform euw1` for a region other than NA.

Riot enforces its limits **per routing value**, so the client keeps a separate
limiter per host: resolving accounts and fetching ranks on `euw1` does not
spend the budget that match downloads need on `europe`.

---

## Limitations

- The bundled model is fit on simulated games. It is calibrated and its
  coefficients are sensibly ordered, but it is a prior, not a measurement of
  the current patch. `harvest` and `train --data` replace it.
- Live-mode team gold is estimated, typically within a few hundred per team.
- `watch` cannot see live gold or kills, because Riot does not expose them.
- Development API keys expire every 24 hours. `doctor` tells you when yours has.
- `spectator-v5/featured-games` is not granted to development keys. `doctor`
  probes `champion-rotations` instead, because a health check built on
  `featured-games` reports a perfectly good key as rejected.
- Summoner's Rift 5v5 only. ARAM and Arena have different dynamics and are not
  modelled.
- Not affiliated with or endorsed by Riot Games.

## Development

```bash
pip install -e .[dev]
pytest                      # 120 tests
python -m pyflakes rift_oracle tests
```

The suite checks the structural guarantees directly: that swapping teams flips
the probability exactly, that attributions sum to the real change with no
residual, that every feature response is non-decreasing, that the advice engine
is symmetric between sides, and that the Riot payload semantics that are easy
to get backwards (`BUILDING_KILL.teamId` is the team that *lost* the building;
`ORDER` is blue) are right.

The client tests run against a fake session, so they never touch the network,
and the whole suite runs against a throwaway state directory so it can neither
read your API key nor be fooled by a previous run's cached responses.

## License

MIT
