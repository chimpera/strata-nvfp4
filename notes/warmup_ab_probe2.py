#!/usr/bin/env python3
"""Warm-up A/B/C v2 — the interrupt must be SUBSTANTIVE.

v1's 5-token interrupt never displaced the main conversation's hot experts
(prefill misses stream without admitting; only decode admits, and 8 decoded
tokens admit nothing). v1 therefore measured the K/V slot machinery, not the
expert-cache pollution the warm-up hypothesis is about. v2 changes exactly
that:

  - each interrupt is a NEW conversation (~1k fresh tokens of an unrelated
    domain: cooking / marine biology / contract law / prosody) so nothing is
    prefix-reused and the prefill feeds the routing/usage ranking,
  - the interrupt GENERATES 250 tokens — decode dispatch is what admits
    experts into the 8,650-slot cache and evicts colder ones (the main's),
  - the main append also generates 250 tokens, so a cold decode ramp
    (misses served from the RAM tier instead of GPU hits) shows in the
    engine log's tok/s and tier lines, not just the wall.

Arms per round r (topic r for both, so the work is identical): U = quiet gap,
I-side = interrupt on the side instance (CUDA0:12340), I-main = interrupt on
the MAIN engine (CUDA1:12341). The measurement is the next main append."""
import json, os, time, urllib.request

MAIN = "http://127.0.0.1:12341/v1/messages"
SIDE = "http://127.0.0.1:12340/v1/messages"
LOG = "/home/guy/serve/services/mlab/state/logs/instances/Qwen_V3.8_Flash_125B_NVFP4_c524_g1_nvfp4.engine.log"
R = "/home/guy/code/strata-nvfp4"

def logoff():
    return os.path.getsize(LOG)

def call(url, model, messages, tag, max_tokens=250):
    body = json.dumps({"model": model, "max_tokens": max_tokens, "messages": messages}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.loads(r.read())
    out = d.get("content", [])
    gen = sum(len(str(b.get("text", ""))) for b in out if isinstance(b, dict))
    print(f"{tag}: {time.time()-t0:.2f}s  in={d['usage']['input_tokens']}  "
          f"gen_chars={gen}  logoff={logoff()}", flush=True)
    return time.time() - t0

# ---------------- unrelated-domain interrupt passages (~1k tokens each)
TOPICS = [
 ("french cooking",
  "You are helping in a professional kitchen. Read this passage carefully.\n\n"
  "A bain-marie holds delicate preparations below the simmer by buffering the "
  "oven's heat through a water bath; custards and terrines set evenly because "
  "the water never exceeds boiling. Carryover heat continues to cook a roast "
  "after it leaves the oven, which is why the loin is pulled five degrees "
  "early and rested under foil. Salt draws moisture through osmosis, so an "
  "early dry brine seasons deeply and dries the skin for browning. The "
  "Maillard reaction between amino acids and reducing sugars builds crust "
  "flavor far above the boiling point, which is why a wet surface steams "
  "instead of searing. Emulsions are suspensions of fat in water stabilized "
  "by lecithin in yolk or proteins in mustard; hollandaise breaks when the "
  "droplets coalesce, and a spoon of water re-forms it. Gluten develops from "
  "kneading, so a tender pastry uses minimal working and cold fat laminated "
  "in layers. Starches gelatinize near the simmer and then retrograde as they "
  "cool, which is why day-old rice fries into distinct grains and gravy "
  "thins on the reheat. Acid brightens at the end of cooking; a slow braise "
  "wants wine added early so the alcohol cooks off and the acidity mellows "
  "into the collagen, which converts to gelatin only after hours just under "
  "the simmer. Yeast dough proofs twice: bulk fermentation develops flavor "
  "through fermentation byproducts, and the shaped proof restores the gas "
  "the shaping knocked out. Oven spring is the final burst of expansion "
  "before the crust sets, helped by steam that keeps the surface flexible. "
  "Carameled sugar passes thread, soft ball, firm ball, hard crack stages as "
  "water leaves and concentration rises; a candy thermometer is really a "
  "moisture meter. Carry a pan off the heat by its handle, not its weight; "
  "season cast iron by polymerizing thin coats of oil; sharpen on a stone "
  "to the burr, then strop it off.\n\n"
  "Now: summarize the passage in one paragraph, then list the three techniques "
  "a new cook most often gets wrong, each with the fix."),
 ("marine biology",
  "You are assisting a marine research station. Read this passage carefully.\n\n"
  "The thermocline marks the boundary between the sun-warmed mixed layer and "
  "the cold deep water beneath; seasonal stratification controls when "
  "nutrients can return to the photic zone. Diatoms bloom where iron and "
  "silicate suffice, building frustules of silica that ballast organic "
  "matter toward the seafloor and make them the great exporters of carbon. "
  "Copepods, not fish, are the most numerous metazoans in the sea, and their "
  "diel vertical migration moves biomass downward every dawn in the largest "
  "synchronized animal movement on the planet. Coral bleaching follows the "
  "expulsion of zooxanthellae under heat stress; the symbiosis fails a degree "
  "or two above the summer maximum the colony is adapted to. Whale falls "
  "pass through scavenger, enrichment, and sulfophilic stages, the last "
  "powered by chemosynthetic bacteria oxidizing sulfide from anaerobic decay, "
  "and they may serve as evolutionary stepping stones for vent fauna. "
  "Tides are long waves forced by the moon's gradient of gravity, Coriolis "
  "turns them into rotating amphidromic systems, and the shape of a basin "
  "decides whether the tide arrives as a progressive wave or a standing "
  "seiche. Lateral line organs sense pressure gradients, letting schooling "
  "fish hold position in vortices shed by their neighbors. Otoliths accrete "
  "daily rings, which is how a flounder's age is read. Many deep-sea fish "
  "use counterillumination, ventral photophores tuned to match faint "
  "downwelling light so their silhouette vanishes from below. Ocean "
  "acidification shifts carbonate chemistry toward bicarbonate, and "
  "calcifying pteropods thin first, rippling into the salmon that eat them. "
  "Kelp anchors with a holdfast, not roots, and gas bladders lift the "
  "fronds into the light; urchin barrens follow when otters, their "
  "predators, are removed. The mesopelagic layer, never sampled by nets "
  "well, likely holds the greatest vertebrate biomass on Earth.\n\n"
  "Now: summarize the passage in one paragraph, then list the three processes "
  "most important for the ocean carbon cycle, each with a one-line reason."),
 ("contract law",
  "You are assisting a contracts paralegal. Read this passage carefully.\n\n"
  "A contract needs offer, acceptance, and consideration; the mirror image "
  "rule made acceptance strict at common law, but the battle of the forms is "
  "now governed by the knockout rule, under which conflicting terms drop out "
  "and gap-fillers apply. Consideration is the bargained-for exchange, and a "
  "mere past consideration or a preexisting duty modified without new "
  "consideration fails, though good-faith modification is honored under the "
  "code. Promissory estoppel enforces reliance on a promise that would "
  "otherwise be gratuitous; it is a shield against injustice, and its reach "
  "beyond that is contested. Interpretation begins with the plain meaning of "
  "the four corners, then course of dealing, course of performance, and "
  "trade usage, in that order of weight under the code. Parol evidence "
  "bars contradicting a fully integrated writing but admits surrounding "
  "circumstances to resolve ambiguity. Conditions precedent control the duty "
  "to perform; substantial performance defeats a condition in construction "
  "contracts where the deviation is not willful, but the perfect tender rule "
  "still governs the sale of goods. Anticipatory repudiation lets the "
  "non-breaching party suspend immediately, though retraction before a "
  "material change in position revives the duty. Damages aim for expectation "
  "interest, limited by certainty, foreseeability under Hadley, and "
  "mitigation; specific performance is reserved where goods are unique or "
  "damages are inadequate. Liquidated damages must be a reasonable estimate "
  "of harm, or they are an unenforceable penalty. The statute of frauds "
  "requires a writing for land, for goods above a threshold, and for "
  "obligations that cannot be performed within a year. An indemnity clause "
  "allocates third-party liability; a limitation-of-liability clause caps "
  "it, and courts construe both against the drafter where ambiguous.\n\n"
  "Now: summarize the passage in one paragraph, then list the three doctrines "
  "a first-year associate most often confuses, each with the distinction."),
 ("metrical poetry",
  "You are assisting a poetry editor. Read this passage carefully.\n\n"
  "Metrical feet are patterns of stressed and unstressed syllables: the iamb "
  "rises, the trochee falls, the anapest and dactyl move in threes. English "
  "verse is accentual-syllabic, counting both stresses and syllables, and "
  "iambic pentameter became its spine because it sits near the length of an "
  "English breath group. Substitution is the engine of life in a meter: a "
  "trochaic inversion at a line head, a feminine ending adding an unstressed "
  "eleventh, a mid-line spondee slowing the voice. Caesura, the mid-line "
  "pause, and enjambment, the run-over line, work against the couplet's "
  "closure; the sonnet's volta at line nine turns argument as much as tone. "
  "Alliterative accentual verse, the older Germanic line, stresses four "
  "syllables with a medial break and binds the halves by initial sound, as "
  "the Old English poets did without rhyme. Syllabic verse counts syllables "
  "but not stresses, as in much French and some Auden; the haiku's five, "
  "seven, five is syllabic and, in Japanese, mora-counting. Skeltonics "
  "tumble in short rhymed lines; the dolnik loosens the iamb until only the "
  "beats are felt, which is how much modern free verse still scans with a "
  "shadow meter. Rhyme schemes carry argument: the couplet epigrammatizes, "
  "the terza rima interlocks forward, the ottava rima ends each stanza in a "
  "punch line. A strong-stress performance tradition, from ballad to rap, "
  "regulates by beat and lets syllables crowd or stretch against it. "
  "Scansion is argument: to stress a syllable the meter leaves weak is to "
  "read the poem's resistance, the way a composer's syncopation is heard "
  "against the count it refuses.\n\n"
  "Now: summarize the passage in one paragraph, then list the three metrical "
  "devices an editor most often has to explain to poets, each with an "
  "example line of your own invention."),
]

# ---------------- build the deep main conversation (~140k, repo code)
msgs = [{"role": "user", "content":
         "You are auditing a C++ inference engine; the following turns carry the "
         "repository's own files. Acknowledge each briefly.\n" * 20}]
FILES = [R + "/src/prefill/prefill.cpp",
         R + "/serve/server.py",
         R + "/src/kernels/cpu/pool.cpp",
         R + "/src/core/expert_cache.cpp",
         R + "/serve/frontend.py",
         R + "/post-restore-slow-tail.md"]
print(f"== PROBE2_START logoff={logoff()}", flush=True)
n_turn = 0
for path in FILES:
    text = open(path, errors="replace").read()
    for off in range(0, len(text), 12000):
        win = text[off:off + 12000]
        if len(win) < 2000:
            break
        n_turn += 1
        msgs.append({"role": "user", "content":
                     f"Keep in context (part {off//12000 + 1}):\n```\n{win}\n```\nAcknowledge briefly."})
        call(MAIN, "CUDA1", msgs, f"build {n_turn:02d} {path.split('/')[-1][:20]}", max_tokens=8)

# appends: a DIFFERENT chunk each round (each is fresh tokens, never reused)
APPEND_FILES = [R + "/AGENTS.md", R + "/docs/DETAILS.md", R + "/README.md", R + "/docs/DETAILS.md"]
appends = []
for i, p in enumerate(APPEND_FILES):
    t = open(p, errors="replace").read()
    appends.append((i, t[i * 1500 % max(len(t) - 2400, 1):][:2400]))

def main_append(round_no):
    i, chunk = appends[round_no % len(appends)]
    msgs.append({"role": "user", "content":
                 "Also keep this:\n```\n" + chunk + "\n```\nAcknowledge briefly, then stop."})
    return call(MAIN, "CUDA1", msgs, f"r{round_no} APPEND file{i}", max_tokens=250)

def interrupt(round_no, target, tag):
    topic, passage = TOPICS[round_no % len(TOPICS)]
    call(target, "CUDA0" if target == SIDE else "CUDA1",
         [{"role": "user", "content": passage}], f"{tag} [{topic}]", max_tokens=250)

CONDS = ["U", "I-side", "I-main"]
URLS = {"I-side": SIDE, "I-main": MAIN}
results = {c: [] for c in CONDS}
for r in range(4):
    for c in CONDS:
        if c != "U":
            interrupt(r, URLS[c], f"  r{r} {c}")
        else:
            time.sleep(7.0)   # a gap of the substantive interrupt's rough wall
        results[c].append(main_append(r))
    CONDS = CONDS[1:] + CONDS[:1]

print("\n== main-append walls (prompt+250 generated tokens) by condition", flush=True)
for c in results:
    v = results[c]
    print(f"{c:7s} mean {sum(v)/len(v):.2f}s   runs " + " ".join(f"{x:.2f}" for x in v), flush=True)
print(f"== PROBE2_END logoff={logoff()}", flush=True)
