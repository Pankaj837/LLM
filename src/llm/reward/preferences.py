"""Synthetic preference data with deliberately conflicting criteria.

Honest framing for the writeup: this is *not* human preference data. It is a
controlled testbed. Each response is generated with two known attributes —
detail level and register — and the preference labels are derived from those
attributes rather than collected from annotators.

That is a real limitation (the model can learn surface cues like sentence count
and contraction frequency instead of anything deeper), but it buys the one
thing needed to prove a *conditional* reward model works: ground truth about
when two criteria should disagree. With scraped preference data you cannot
tell whether a ranking flip is the model working correctly or the model being
noisy. Here you can.

Three conditions:
    helpful  -> prefers the more detailed response
    concise  -> prefers the shorter response
    formal   -> prefers the formal register

helpful and concise are constructed to be in direct opposition, so any pair
differing in detail is a flip-test pair for that pairing.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Tuple

CONDITIONS = ["helpful", "concise", "formal"]
COND_TO_ID = {c: i for i, c in enumerate(CONDITIONS)}

# ---------------------------------------------------------------- content

# (question, [(casual_fact, formal_fact), ...])
TOPICS: List[Tuple[str, List[Tuple[str, str]]]] = [
    ("How do I reset my password?", [
        ("Just click the forgot password link on the login page.",
         "Please select the password recovery option on the login page."),
        ("You'll get an email with a reset link in a minute or two.",
         "A message containing a reset link will be delivered shortly."),
        ("If it doesn't show up, check your spam folder.",
         "Should the message not arrive, please review your spam folder."),
    ]),
    ("What's the best way to learn programming?", [
        ("Pick one language and build stuff with it.",
         "It is advisable to select a single language and construct projects with it."),
        ("Reading docs helps way more than watching tutorials.",
         "Consulting documentation is considerably more effective than viewing tutorials."),
        ("Try to code a bit every day rather than cramming.",
         "Regular daily practice is preferable to concentrated study sessions."),
    ]),
    ("Why is my laptop running slow?", [
        ("Probably too many programs open at once.",
         "The likely cause is an excessive number of concurrent applications."),
        ("Check how much disk space you've got left.",
         "Please verify the amount of available disk storage."),
        ("Restarting it clears out a surprising amount of junk.",
         "A system restart will clear a substantial quantity of temporary data."),
    ]),
    ("How do I make better coffee at home?", [
        ("Grind the beans right before you brew.",
         "The beans should be ground immediately prior to brewing."),
        ("Water temperature matters a lot, aim just off the boil.",
         "Water temperature is significant; it should be slightly below boiling."),
        ("Weigh your coffee instead of using scoops.",
         "Measurement by weight is preferable to volumetric scoops."),
    ]),
    ("What should I look for when renting an apartment?", [
        ("Check the water pressure and heating before you sign.",
         "Please inspect the water pressure and heating prior to signing."),
        ("Ask what's actually included in the rent.",
         "Enquire as to which utilities are included in the stated rent."),
        ("Visit at night to see what the area's really like.",
         "An evening visit is recommended to assess the neighbourhood accurately."),
    ]),
    ("How can I improve my sleep?", [
        ("Keep your bedtime the same every night.",
         "A consistent nightly bedtime should be maintained."),
        ("Screens before bed really do mess things up.",
         "Screen exposure prior to sleep has a documented adverse effect."),
        ("A cooler room usually helps you drop off faster.",
         "A reduced ambient temperature typically shortens sleep onset."),
    ]),
    ("What's a good first houseplant?", [
        ("Go with a pothos, they're basically unkillable.",
         "A pothos is recommended, as it tolerates considerable neglect."),
        ("Don't water it until the soil's dry a couple inches down.",
         "Water only once the upper soil layer has thoroughly dried."),
        ("Indirect light works fine, no need for a sunny window.",
         "Indirect illumination is sufficient; direct sunlight is unnecessary."),
    ]),
    ("How do I start running?", [
        ("Start with a walk-run mix, don't just sprint off.",
         "Commence with alternating intervals of walking and running."),
        ("Good shoes are worth the money here.",
         "Appropriate footwear represents a worthwhile investment."),
        ("Add distance slowly or you'll pick up an injury.",
         "Distance should be increased gradually to avoid injury."),
    ]),
    ("Why won't my code compile?", [
        ("Read the first error, the rest are usually knock-on effects.",
         "Please address the first reported error; subsequent errors are often consequential."),
        ("Missing semicolons and brackets cause most of these.",
         "Absent semicolons and unbalanced brackets account for the majority of such failures."),
        ("Check you actually saved the file.",
         "Please confirm that the file has been saved."),
    ]),
    ("How should I prepare for a job interview?", [
        ("Look up the company and have a few questions ready.",
         "Research the organisation and prepare several questions in advance."),
        ("Practice saying your experience out loud, it's harder than it sounds.",
         "Rehearse your experience verbally; articulation is more difficult than anticipated."),
        ("Get there early so you're not flustered.",
         "Arrive in advance of the scheduled time to avoid unnecessary stress."),
    ]),
    ("What's the difference between RAM and storage?", [
        ("RAM is short-term, storage keeps things after shutdown.",
         "RAM provides temporary capacity; storage retains data following shutdown."),
        ("More RAM helps with lots of tabs open.",
         "Additional RAM improves performance with numerous concurrent applications."),
        ("Storage size is about how much stuff you can keep.",
         "Storage capacity determines the volume of retained data."),
    ]),
    ("How do I keep my bike in good shape?", [
        ("Keep the tyres pumped up properly.",
         "Tyre pressure should be maintained at the specified level."),
        ("Oil the chain every few weeks.",
         "The chain requires lubrication at intervals of several weeks."),
        ("Wipe it down after riding in the rain.",
         "The frame should be dried following use in wet conditions."),
    ]),
    ("What's a reasonable budget for groceries?", [
        ("Depends on where you live, but track it for a month first.",
         "This varies by location; a month of expenditure tracking is advised."),
        ("Cooking in bulk brings the cost down a lot.",
         "Batch preparation substantially reduces overall expenditure."),
        ("Buying seasonal produce is cheaper than you'd expect.",
         "Seasonal produce is considerably more economical than anticipated."),
    ]),
    ("How do I take better photos?", [
        ("Get closer to your subject than feels natural.",
         "Position yourself nearer to the subject than instinct suggests."),
        ("Shoot in the hour after sunrise or before sunset.",
         "Photograph during the hour following sunrise or preceding sunset."),
        ("Watch what's in the background, it ruins more shots than anything.",
         "Attend to background composition, a frequent cause of unsatisfactory results."),
    ]),
    ("Should I learn a second language?", [
        ("Yeah, and pick one you'll actually get to use.",
         "This is advisable; select a language you will have occasion to use."),
        ("Twenty minutes daily beats a long session once a week.",
         "Twenty minutes of daily study surpasses a single weekly session."),
        ("Speaking early feels awkward but speeds everything up.",
         "Early conversational practice, though uncomfortable, accelerates progress."),
    ]),
    ("How do I organise my files better?", [
        ("Pick one folder structure and actually stick to it.",
         "Adopt a single folder structure and adhere to it consistently."),
        ("Dates in filenames make sorting way easier.",
         "The inclusion of dates in filenames considerably simplifies sorting."),
        ("Clear out downloads once a month.",
         "The downloads directory should be cleared monthly."),
    ]),
    ("What's worth knowing before adopting a cat?", [
        ("They need way less attention than dogs but not none.",
         "Cats require less attention than dogs, though not none whatsoever."),
        ("Vet bills add up, budget for them.",
         "Veterinary costs accumulate and should be budgeted for."),
        ("Get a scratching post before they find your sofa.",
         "Provide a scratching post in advance of furniture damage."),
    ]),
    ("How do I write a better email?", [
        ("Put the ask in the first line.",
         "The request should appear in the opening sentence."),
        ("Shorter is nearly always better.",
         "Brevity is almost invariably preferable."),
        ("Read it back once before sending.",
         "Review the message once prior to transmission."),
    ]),
    ("What causes a headache after screen work?", [
        ("Usually eye strain from not blinking enough.",
         "Ocular strain from reduced blink frequency is the usual cause."),
        ("Look at something far away every twenty minutes.",
         "Focus on a distant object at twenty minute intervals."),
        ("Bad posture at the desk feeds into it too.",
         "Suboptimal desk posture is a contributing factor."),
    ]),
    ("How do I save money on travel?", [
        ("Book flights midweek, they're usually cheaper.",
         "Midweek flight bookings are generally more economical."),
        ("Travelling just outside peak season saves a fortune.",
         "Travel immediately outside peak season yields substantial savings."),
        ("Trains beat flights on short hops once you count airport time.",
         "Rail transport surpasses air travel on short routes when transfer time is included."),
    ]),
]


@dataclass
class Response:
    text: str
    detail: int     # 1 = brief, 3 = detailed
    formal: int     # 0 = casual, 1 = formal


def build_responses(facts: List[Tuple[str, str]]) -> List[Response]:
    """Four variants per topic: {brief, detailed} x {casual, formal}."""
    out = []
    for formal in (0, 1):
        for detail in (1, 3):
            parts = [f[formal] for f in facts[:detail]]
            out.append(Response(" ".join(parts), detail=detail, formal=formal))
    return out


def prefers(cond: str, a: Response, b: Response) -> int | None:
    """Which response a criterion prefers: 0 for a, 1 for b, None if tied."""
    if cond == "helpful":
        if a.detail == b.detail:
            return None
        return 0 if a.detail > b.detail else 1
    if cond == "concise":
        if a.detail == b.detail:
            return None
        return 0 if a.detail < b.detail else 1
    if cond == "formal":
        if a.formal == b.formal:
            return None
        return 0 if a.formal > b.formal else 1
    raise ValueError(cond)


@dataclass
class PreferencePair:
    prompt: str
    chosen: str
    rejected: str
    condition: str
    conflicting: bool   # True if some other condition ranks this pair oppositely


def build_dataset(seed: int = 0) -> List[PreferencePair]:
    rng = random.Random(seed)
    pairs: List[PreferencePair] = []

    for prompt, facts in TOPICS:
        variants = build_responses(facts)
        for i in range(len(variants)):
            for j in range(i + 1, len(variants)):
                a, b = variants[i], variants[j]
                for cond in CONDITIONS:
                    winner = prefers(cond, a, b)
                    if winner is None:
                        continue
                    # Flag pairs where the criteria genuinely disagree — these
                    # are the ones that a non-conditional reward model
                    # provably cannot get right for every criterion at once.
                    others = [prefers(c, a, b) for c in CONDITIONS if c != cond]
                    conflicting = any(o is not None and o != winner for o in others)
                    chosen, rejected = (a, b) if winner == 0 else (b, a)
                    pairs.append(PreferencePair(
                        prompt=prompt, chosen=chosen.text, rejected=rejected.text,
                        condition=cond, conflicting=conflicting,
                    ))

    rng.shuffle(pairs)
    return pairs


def split(pairs: List[PreferencePair], val_frac: float = 0.2, seed: int = 0):
    """Split by *topic*, not by pair.

    Splitting randomly over pairs would put the same prompt and the same
    response strings on both sides, so validation accuracy would measure
    memorisation rather than generalisation to unseen prompts.
    """
    rng = random.Random(seed)
    prompts = sorted({p.prompt for p in pairs})
    rng.shuffle(prompts)
    n_val = max(1, int(len(prompts) * val_frac))
    val_prompts = set(prompts[:n_val])
    train = [p for p in pairs if p.prompt not in val_prompts]
    val = [p for p in pairs if p.prompt in val_prompts]
    return train, val


def condition_blind_ceiling(pairs: List[PreferencePair]) -> float:
    """Best accuracy achievable by any reward model that ignores the condition.

    Such a model produces one score per response, so for a given pair it must
    commit to a single ranking regardless of criterion. Where two criteria
    disagree it is guaranteed to be wrong on at least one. This computes that
    bound exactly, which turns "the conditional model is better" from a vague
    claim into a threshold the model either clears or does not.
    """
    from collections import defaultdict
    groups = defaultdict(list)
    for p in pairs:
        groups[(p.prompt, frozenset([p.chosen, p.rejected]))].append(p.chosen)
    best = sum(max(v.count(c) for c in set(v)) for v in groups.values())
    return best / max(len(pairs), 1)


def swap_pair_for(prompt: str) -> Tuple[str, str]:
    """(detailed, brief) casual-register responses for a topic, for the condition-swap test."""
    facts = dict(TOPICS)[prompt]
    variants = build_responses(facts)
    detailed = next(v for v in variants if v.detail == 3 and v.formal == 0).text
    brief = next(v for v in variants if v.detail == 1 and v.formal == 0).text
    return detailed, brief


def surface_heuristic_accuracy(pairs: List[PreferencePair]) -> float:
    """Accuracy of a hand-written, ~5-line rule using only character length and
    contraction count. This is the bar a *learned* reward model must clear to
    show it has learned anything beyond the synthetic data's surface cues."""
    import re

    def contractions(t: str) -> int:
        return len(re.findall(r"'(?:re|ve|ll|s|t|m|d)\b", t))

    def formality(t: str):
        words = t.split()
        return (-contractions(t), sum(map(len, words)) / max(len(words), 1))

    def picks_chosen(p: PreferencePair) -> bool:
        if p.condition == "helpful":
            return len(p.chosen) > len(p.rejected)
        if p.condition == "concise":
            return len(p.chosen) < len(p.rejected)
        return formality(p.chosen) > formality(p.rejected)

    return sum(picks_chosen(p) for p in pairs) / max(len(pairs), 1)


def corpus_for_tokenizer() -> List[str]:
    """Every string the reward model will ever see, for BPE training."""
    lines = []
    for prompt, facts in TOPICS:
        lines.append(prompt)
        for r in build_responses(facts):
            lines.append(r.text)
    return lines


if __name__ == "__main__":
    pairs = build_dataset()
    tr, va = split(pairs)
    n_conf = sum(p.conflicting for p in pairs)
    print(f"{len(pairs)} pairs ({n_conf} conflicting), train={len(tr)} val={len(va)}")
    for c in CONDITIONS:
        print(f"  {c:8s}: {sum(p.condition == c for p in pairs)}")
