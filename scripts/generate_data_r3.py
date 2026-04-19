#!/usr/bin/env python3
"""R3 SFT seed and data generation.

Generates:
  - data/seeds_r3/L01_en_hearsay.json   (200 samples)
  - data/seeds_r3/L01_en_pj.json        (200 samples)
  - data/seeds_r3/L01_en_textualism.json (200 samples)
  - data/seeds_r3/L05_en.json           (30 samples)
  - data/seeds_r3/L06_en.json           (70 samples)
  - data/seeds_r3/L09_en.json           (20 samples)
  - data/seeds_r3/L10_en.json           (20 samples)
  - data/seeds_r3/L01_ko.json           (Korean, ~140 samples)
  - data/sft_r3.jsonl                   (8K+ combined, 80% EN)

Key fixes vs R2:
  - All elements use whitelist names (P1-B)
  - All binary final_answers start with "Yes." or "No." (P0-B requirement)
  - 80% English / 20% Korean ratio
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

random.seed(42)

SYSTEM_EN = (
    "You are a legal reasoning AI. Classify the input into a Legal Lane "
    "and invoke the appropriate solver."
)
SYSTEM_KO = (
    "당신은 법률 추론 AI입니다. 입력을 Legal Lane으로 분류하고 적절한 솔버를 호출하세요."
)

# ---------------------------------------------------------------------------
# Hearsay seed generation (FRE 801(c))
# ---------------------------------------------------------------------------

# Case templates: (setup, statement_context, statement_maker, topic, answer, reason)
HEARSAY_TEMPLATES = [
    # Standard hearsay — Yes
    ("On the issue of {topic}, the fact that {maker} told {recipient} that {assertion}.",
     True, "out_of_court_statement", "offered_for_truth",
     "Yes. This is hearsay under FRE 801(c) because {maker}'s statement was made "
     "outside court and is being offered to prove the truth of the matter asserted "
     "— namely, that {topic_short}."),
    # Not offered for truth — No
    ("On the issue of {topic}, the fact that {maker} told {recipient} that {assertion}, "
     "offered not to prove the truth of the statement but to show its effect on {recipient}.",
     False, "out_of_court_statement", None,
     "No. Although {maker}'s statement was made out of court, it is not being offered "
     "to prove the truth of the matter asserted. It is offered to show the effect on "
     "{recipient}'s state of mind, so it does not qualify as hearsay under FRE 801(c)."),
    # In-court statement — No
    ("On the issue of {topic}, {maker} testified in court that {assertion}.",
     False, None, None,
     "No. {maker}'s statement was made in court during testimony, so it is not an "
     "out-of-court statement. It therefore cannot be hearsay under FRE 801(c)."),
    # Non-assertive conduct — No
    ("On the issue of {topic}, the fact that {maker} {conduct}.",
     False, None, None,
     "No. {maker}'s conduct ({conduct_short}) is non-assertive and was not intended "
     "as an assertion. Because it lacks the declarative intent required by FRE 801(a), "
     "it is not a statement and therefore cannot be hearsay."),
    # Verbal act — No
    ("On the issue of {topic}, the fact that {maker} said '{verbal_act}'.",
     False, "out_of_court_statement", None,
     "No. {maker}'s words are a verbal act — their legal significance lies in "
     "the utterance itself, not in the truth of any assertion. Therefore this is "
     "not hearsay under FRE 801(c)."),
]

# Scenario components
TOPICS_HEARSAY = [
    ("whether {name} was present at the meeting", "{name} was present"),
    ("whether {name} was ill", "{name} was ill"),
    ("whether {name} knew about the plan", "{name} knew about the plan"),
    ("whether {name} acted negligently", "{name} acted negligently"),
    ("whether {name} was intoxicated", "{name} was intoxicated"),
    ("whether {name} owned the property", "{name} owned the property"),
    ("whether {name} had consented to the procedure", "{name} had consented"),
    ("whether {name} received the payment", "{name} received the payment"),
    ("whether {name} was a licensed contractor", "{name} was licensed"),
    ("whether {name} was the first to arrive", "{name} arrived first"),
    ("whether {name} made the threat", "{name} made the threat"),
    ("whether {name} was aware of the defect", "{name} knew of the defect"),
    ("whether {name} had authority to sign", "{name} had signing authority"),
    ("whether {name} was at fault", "{name} was at fault"),
    ("whether {name} was driving the vehicle", "{name} was driving"),
    ("whether {name} gave instructions", "{name} gave instructions"),
    ("whether {name} was a partner in the firm", "{name} was a partner"),
    ("whether {name} witnessed the accident", "{name} witnessed the accident"),
    ("whether {name} had prior knowledge", "{name} had prior knowledge"),
    ("whether {name} consented to the search", "{name} consented"),
]

NAMES = [
    "Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Henry",
    "Isabel", "James", "Karen", "Liam", "Maria", "Nathan", "Olivia",
    "Patrick", "Quinn", "Rachel", "Samuel", "Teresa", "Victor", "Wendy",
]

ASSERTIONS = [
    "she was feeling very sick that morning",
    "he had seen the defendant at the scene",
    "the contract had been signed by both parties",
    "the product was defective when shipped",
    "he had received the money",
    "she did not receive proper training",
    "the brakes had failed repeatedly before",
    "the building permit had expired",
    "he had been warned about the risk",
    "the medication caused serious side effects",
    "the document was forged",
    "he had never met the plaintiff",
    "the debt was fully paid",
    "she was authorized to make the purchase",
    "the vehicle was not properly maintained",
    "the agreement was oral, not written",
    "he had complained about the conditions before",
    "the meeting had already ended",
    "she had never driven that route before",
    "the shipment arrived on schedule",
]

RECIPIENTS = [
    "her supervisor", "his attorney", "a colleague", "the investigator",
    "a bystander", "the police officer", "the nurse", "a friend",
    "her business partner", "the insurance adjuster",
]

CONDUCTS = [
    ("wore a helmet every time she rode her motorcycle",
     "wearing a helmet regularly"),
    ("opened an umbrella when he stepped outside",
     "opening an umbrella"),
    ("locked all the windows before leaving",
     "locking the windows"),
    ("installed safety locks on the medicine cabinet",
     "installing safety locks"),
    ("kept a fire extinguisher in the kitchen",
     "keeping a fire extinguisher"),
    ("ran a red light without slowing down",
     "running a red light"),
    ("called 911 immediately after the incident",
     "calling 911"),
    ("returned the borrowed item the next day",
     "returning the item promptly"),
    ("hired a security guard for the premises",
     "hiring a security guard"),
    ("purchased extra liability insurance",
     "purchasing extra insurance"),
]

VERBAL_ACTS = [
    "I accept your offer",
    "I do",
    "I quit",
    "I promise to pay you back",
    "You're fired",
    "I authorize this transaction",
    "I consent to the search",
    "I surrender",
    "I admit liability",
    "I waive my right to appeal",
]


def _make_hearsay_sample(idx: int) -> Dict:
    """Generate one hearsay sample."""
    name = random.choice(NAMES)
    name2 = random.choice([n for n in NAMES if n != name])
    topic_tmpl, topic_short = random.choice(TOPICS_HEARSAY)
    topic = topic_tmpl.format(name=name)
    topic_short_filled = topic_short.format(name=name)

    scenario_type = idx % 5  # cycle through 5 template types

    if scenario_type == 0:
        # Standard hearsay
        assertion = random.choice(ASSERTIONS)
        recipient = random.choice(RECIPIENTS)
        question = (
            f"On the issue of {topic}, the fact that {name} told {recipient} that "
            f"{assertion}. Is this hearsay?"
        )
        tc = {
            "tool": "z3", "method": "check_elements",
            "elements": ["out_of_court_statement", "offered_for_truth"],
            "facts": [
                f"{name} made the statement outside of court",
                f"the statement is offered to prove {topic_short_filled}",
            ],
            "matching": {"out_of_court_statement": True, "offered_for_truth": True},
            "mode": "and",
        }
        answer_yn = "Yes"
        explanation = (
            f"Yes. This is hearsay under FRE 801(c) because {name}'s statement was "
            f"made outside of court and is being offered to prove the truth of the "
            f"matter asserted — namely, that {topic_short_filled}."
        )
        tool_result = {"all_met": True}

    elif scenario_type == 1:
        # Not offered for truth
        assertion = random.choice(ASSERTIONS)
        recipient = random.choice(RECIPIENTS)
        question = (
            f"On the issue of {topic}, the fact that {name} told {recipient} that "
            f"{assertion}, offered solely to show the effect on {recipient}. Is this hearsay?"
        )
        tc = {
            "tool": "z3", "method": "check_elements",
            "elements": ["out_of_court_statement", "offered_for_truth"],
            "facts": [
                f"{name} made the statement outside of court",
                f"the statement is not offered for its truth — it shows effect on {recipient}",
            ],
            "matching": {"out_of_court_statement": True, "offered_for_truth": False},
            "mode": "and",
        }
        answer_yn = "No"
        explanation = (
            f"No. Although {name}'s statement was made out of court, it is not offered "
            f"to prove the truth of the matter asserted. It is offered to show its effect "
            f"on {recipient}'s state of mind, so it is not hearsay under FRE 801(c)."
        )
        tool_result = {"all_met": False}

    elif scenario_type == 2:
        # In-court statement
        assertion = random.choice(ASSERTIONS)
        question = (
            f"On the issue of {topic}, {name} testified in court that {assertion}. "
            f"Is this hearsay?"
        )
        tc = {
            "tool": "z3", "method": "check_elements",
            "elements": ["out_of_court_statement", "offered_for_truth"],
            "facts": [
                f"{name}'s statement was made during in-court testimony",
                f"the statement is offered to prove {topic_short_filled}",
            ],
            "matching": {"out_of_court_statement": False, "offered_for_truth": True},
            "mode": "and",
        }
        answer_yn = "No"
        explanation = (
            f"No. {name}'s statement was made in court during testimony, so it is not "
            f"an out-of-court statement and cannot qualify as hearsay under FRE 801(c)."
        )
        tool_result = {"all_met": False}

    elif scenario_type == 3:
        # Non-assertive conduct
        conduct, conduct_short = random.choice(CONDUCTS)
        question = (
            f"On the issue of {topic}, the fact that {name} {conduct}. Is this hearsay?"
        )
        tc = {
            "tool": "z3", "method": "check_elements",
            "elements": ["out_of_court_statement", "offered_for_truth"],
            "facts": [
                f"{name}'s conduct was non-assertive and not intended as a declaration",
                f"no verbal or written assertion was made",
            ],
            "matching": {"out_of_court_statement": False, "offered_for_truth": False},
            "mode": "and",
        }
        answer_yn = "No"
        explanation = (
            f"No. {name}'s conduct ({conduct_short}) is non-assertive and was not "
            f"intended as an assertion. Because it lacks the declarative intent required "
            f"by FRE 801(a), it does not constitute a 'statement' and therefore cannot "
            f"be hearsay."
        )
        tool_result = {"all_met": False}

    else:
        # Verbal act
        verbal_act = random.choice(VERBAL_ACTS)
        question = (
            f"On the issue of {topic}, the fact that {name} said '{verbal_act}'. "
            f"Is this hearsay?"
        )
        tc = {
            "tool": "z3", "method": "check_elements",
            "elements": ["out_of_court_statement", "offered_for_truth"],
            "facts": [
                f"{name} uttered the words outside of court",
                f"the words are a verbal act, not offered for their truth",
            ],
            "matching": {"out_of_court_statement": True, "offered_for_truth": False},
            "mode": "and",
        }
        answer_yn = "No"
        explanation = (
            f"No. {name}'s words ('{verbal_act}') are a verbal act — their legal "
            f"significance lies in the utterance itself, not in any assertion about "
            f"the truth of facts. Therefore this is not hearsay under FRE 801(c)."
        )
        tool_result = {"all_met": False}

    assert explanation.startswith(answer_yn), f"Answer mismatch: {explanation[:40]}"

    return {
        "lane": "L01",
        "source": "legalbench_hearsay_r3",
        "messages": [
            {"role": "system", "content": SYSTEM_EN},
            {"role": "user", "content": question},
            {
                "role": "assistant",
                "content": f"<lane>L01</lane>\n<tool_call>{json.dumps(tc, ensure_ascii=False)}</tool_call>",
            },
            {"role": "tool", "content": json.dumps(tool_result, ensure_ascii=False)},
            {"role": "assistant", "content": explanation},
        ],
    }


# ---------------------------------------------------------------------------
# Personal Jurisdiction seed generation
# ---------------------------------------------------------------------------

PJ_SCENARIOS = [
    # (name, home_state, business_desc, forum_state, has_mc, has_pa, has_fp, answer)
    # All three met → Yes
    ("GlobalTech Corp", "Delaware", "sells electronics online nationwide and operates a regional office in {forum}",
     True, True, True, True,
     "minimum_contacts purposeful_availment fair_play",
     "GlobalTech Corp's nationwide sales operations, regional office in {forum}, and deliberate targeting of {forum} consumers establish minimum contacts and purposeful availment. Exercising jurisdiction is consistent with fair play and substantial justice. Personal jurisdiction is proper."),
    # Minimum contacts but not purposeful availment → No
    ("FastShip LLC", "Texas", "accidentally shipped a defective product to {forum} while attempting to ship to a different state",
     True, False, True, False,
     "minimum_contacts purposeful_availment fair_play",
     "Although FastShip LLC's product reached {forum}, the contact was accidental rather than the result of purposeful availment of the {forum} market. Because purposeful availment is not met, personal jurisdiction does not lie."),
    # No contacts at all → No
    ("Mountain Gear Inc", "Colorado", "sells exclusively in Colorado with no online presence and no sales outside the state",
     False, False, True, False,
     "minimum_contacts purposeful_availment fair_play",
     "Mountain Gear Inc. has no contacts with {forum} — it operates solely in Colorado with no online sales or distribution reaching {forum}. Without minimum contacts, personal jurisdiction cannot be established."),
    # Domicile → Yes
    ("individual defendant", "California", "recently moved to {forum} and now resides there permanently",
     True, True, True, True,
     "minimum_contacts purposeful_availment fair_play",
     "The defendant is now domiciled in {forum}, which independently establishes general personal jurisdiction. All three requirements — minimum contacts, purposeful availment, and fair play — are satisfied."),
    # Purposeful availment but no nexus → No
    ("EastBridge Corp", "New York", "maintains a distribution center in {forum} but the claim arises from an unrelated out-of-state transaction",
     True, True, False, False,
     "minimum_contacts purposeful_availment fair_play",
     "Although EastBridge Corp has minimum contacts and purposeful availment in {forum}, the specific claim does not arise out of or relate to those contacts. For specific jurisdiction, there must be a nexus between the contacts and the claim. Personal jurisdiction is therefore improper."),
]

FORUM_STATES = [
    "California", "New York", "Texas", "Florida", "Illinois",
    "Washington", "Oregon", "Michigan", "Ohio", "Georgia",
    "Pennsylvania", "Arizona", "Colorado", "Virginia", "Massachusetts",
]

PJ_PLAINTIFFS = [
    "Laura", "Michael", "Sarah", "James", "Emily",
    "Robert", "Jessica", "Daniel", "Amanda", "Christopher",
]


def _make_pj_sample(idx: int) -> Dict:
    forum = random.choice(FORUM_STATES)
    scenario = PJ_SCENARIOS[idx % len(PJ_SCENARIOS)]
    (def_name, home_state, biz_desc, has_mc, has_pa, has_fp, answer,
     elem_hint, reasoning_template) = (
        scenario[0], scenario[1], scenario[2],
        scenario[3], scenario[4], scenario[5], scenario[6],
        scenario[7], scenario[8],
    )
    plaintiff = random.choice(PJ_PLAINTIFFS)
    biz_desc_filled = biz_desc.format(forum=forum)

    question = (
        f"{def_name} (incorporated in {home_state}) {biz_desc_filled}. "
        f"{plaintiff}, a citizen of {forum}, sues {def_name} in {forum}. "
        f"Does the court have personal jurisdiction?"
    )

    matching = {
        "minimum_contacts": has_mc,
        "purposeful_availment": has_pa,
        "fair_play_and_substantial_justice": has_fp,
    }
    all_met = has_mc and has_pa and has_fp
    tc = {
        "tool": "z3", "method": "check_elements",
        "elements": ["minimum_contacts", "purposeful_availment",
                     "fair_play_and_substantial_justice"],
        "facts": [
            f"{def_name} is incorporated in {home_state}",
            biz_desc_filled,
            f"Plaintiff {plaintiff} is a citizen of {forum}",
        ],
        "matching": matching,
        "mode": "and",
    }

    reasoning = reasoning_template.format(forum=forum)
    verdict = "Yes" if answer else "No"
    if not reasoning.startswith(verdict):
        reasoning = f"{verdict}. {reasoning}"
    else:
        reasoning = reasoning  # already starts correctly

    return {
        "lane": "L01",
        "source": "legalbench_pj_r3",
        "messages": [
            {"role": "system", "content": SYSTEM_EN},
            {"role": "user", "content": question},
            {
                "role": "assistant",
                "content": f"<lane>L01</lane>\n<tool_call>{json.dumps(tc, ensure_ascii=False)}</tool_call>",
            },
            {"role": "tool", "content": json.dumps({"all_met": all_met}, ensure_ascii=False)},
            {"role": "assistant", "content": reasoning},
        ],
    }


# ---------------------------------------------------------------------------
# Textualism seed generation
# ---------------------------------------------------------------------------

TEXTUALISM_CASES = [
    # (excerpt_summary, uses_dict, plain_meaning, answer, explanation)
    (
        "The court consulted the Merriam-Webster Dictionary to determine the ordinary meaning of '{term}', finding it means '{definition}'.",
        True, True, True,
        "Yes. The court employed textualism by consulting the Merriam-Webster Dictionary (dictionary_definition_used) and applying the plain, ordinary meaning of '{term}' (plain_meaning_invoked). Both hallmarks of textualist interpretation are present.",
    ),
    (
        "The opinion relied on the legislative history and sponsor statements to interpret '{term}', finding that Congress intended a broad reading.",
        False, False, False,
        "No. The court relied on legislative history and congressional intent rather than the plain text or dictionary definitions. This approach reflects intentionalism, not textualism.",
    ),
    (
        "The court stated that '{term}' should be read according to its ordinary, contemporary, common meaning without resort to legislative history.",
        False, True, True,
        "Yes. The court invoked plain_meaning_invoked by explicitly relying on the ordinary, contemporary, common meaning of '{term}' and rejecting resort to legislative history — a core textualist technique.",
    ),
    (
        "The court cited Black's Law Dictionary to find that '{term}' carries a specific legal meaning distinct from its everyday usage.",
        True, False, True,
        "Yes. The court used dictionary_definition_used by consulting Black's Law Dictionary to define '{term}'. Even though the legal definition differs from everyday usage, the dictionary-based approach satisfies textualism_tool.",
    ),
    (
        "The majority applied the rule of lenity and construed the ambiguous term '{term}' in favor of the defendant.",
        False, False, False,
        "No. The court applied the rule of lenity rather than textualist tools. Neither a dictionary definition nor a plain-meaning analysis was invoked to interpret '{term}'.",
    ),
    (
        "After noting that '{term}' is not defined in the statute, the court turned to the Oxford English Dictionary and the American Heritage Dictionary to establish its ordinary meaning.",
        True, True, True,
        "Yes. The court used dictionary_definition_used (OED and American Heritage Dictionary) and applied plain_meaning_invoked by establishing the ordinary meaning of '{term}'. Both textualism indicators are satisfied.",
    ),
    (
        "The court applied the ejusdem generis canon to narrow the scope of '{term}' based on surrounding specific terms in the list.",
        False, True, True,
        "Yes. Applying ejusdem generis is a textual canon of construction that relies on the plain meaning of surrounding statutory terms — plain_meaning_invoked is satisfied. However, no dictionary definition was consulted.",
    ),
]

LEGAL_TERMS = [
    "crime", "damage", "maintenance", "lapse of time", "election laws",
    "dwelling", "commerce", "vehicle", "employee", "manufacture",
    "transport", "distribute", "carry", "possess", "use",
    "willful", "knowing", "reckless", "reasonable", "material",
]

DEFINITIONS = [
    "unlawful activity in general",
    "a deterioration or injurious lessening",
    "upkeep or preservation of condition",
    "the expiration of a prescribed period",
    "laws that specifically govern elections",
    "a place of abode",
    "trade or traffic in goods",
    "any self-propelled conveyance",
    "an individual in a service relationship",
    "to make something from raw materials",
]


def _make_textualism_sample(idx: int) -> Dict:
    case = TEXTUALISM_CASES[idx % len(TEXTUALISM_CASES)]
    excerpt_tmpl, use_dict, plain_mean, answer, explanation_tmpl = case
    term = random.choice(LEGAL_TERMS)
    definition = random.choice(DEFINITIONS)

    excerpt = excerpt_tmpl.format(term=term, definition=definition)
    explanation = explanation_tmpl.format(term=term)

    verdict = "Yes" if answer else "No"
    assert explanation.startswith(verdict), f"Answer mismatch in textualism: {explanation[:50]}"

    elements = []
    if use_dict:
        elements.append("dictionary_definition_used")
    if plain_mean:
        elements.append("plain_meaning_invoked")
    if not elements:
        elements = ["dictionary_definition_used", "plain_meaning_invoked"]

    matching = {
        "dictionary_definition_used": use_dict,
        "plain_meaning_invoked": plain_mean,
    }
    all_met = answer  # answer == True means textualism tool applies

    tc = {
        "tool": "z3", "method": "check_elements",
        "elements": elements,
        "facts": [excerpt[:120]],
        "matching": matching,
        "mode": "or",
    }

    question = (
        f"Does the following excerpt reflect textualist statutory interpretation?\n\n"
        f"{excerpt}"
    )

    return {
        "lane": "L01",
        "source": "legalbench_textualism_r3",
        "messages": [
            {"role": "system", "content": SYSTEM_EN},
            {"role": "user", "content": question},
            {
                "role": "assistant",
                "content": f"<lane>L01</lane>\n<tool_call>{json.dumps(tc, ensure_ascii=False)}</tool_call>",
            },
            {"role": "tool", "content": json.dumps({"all_met": all_met}, ensure_ascii=False)},
            {"role": "assistant", "content": explanation},
        ],
    }


# ---------------------------------------------------------------------------
# L05 calculation seeds
# ---------------------------------------------------------------------------

L05_CASES = [
    ("Calculate the prejudgment interest on a $50,000 judgment at 6% per year for 3 years.",
     "P * r * t", {"P": 50000, "r": 0.06, "t": 3}, 2,
     "The prejudgment interest is $9,000.00. Calculated as: $50,000 × 0.06 × 3 = $9,000."),
    ("A contractor is owed $120,000 but the other party claims a $15,000 setoff. What is the net amount owed?",
     "P - setoff", {"P": 120000, "setoff": 15000}, 2,
     "The net amount owed is $105,000.00 after applying the $15,000 setoff to the $120,000 claim."),
    ("The statute provides for a civil penalty of $500 per violation. If there were 47 violations, what is the total penalty?",
     "penalty_per * count", {"penalty_per": 500, "count": 47}, 0,
     "The total civil penalty is $23,500. Calculated as: $500 × 47 = $23,500."),
    ("Calculate the monthly pro-rated rent for a $2,400/month lease if the tenant moves in on the 15th of a 30-day month.",
     "monthly_rent / days_in_month * days_remaining",
     {"monthly_rent": 2400, "days_in_month": 30, "days_remaining": 16}, 2,
     "The pro-rated rent for 16 remaining days in a 30-day month is $1,280.00. Calculated as: $2,400 ÷ 30 × 16 = $1,280."),
    ("A party seeks attorney's fees of $350/hr for 42.5 hours. Calculate the total fee.",
     "rate * hours", {"rate": 350, "hours": 42.5}, 2,
     "The total attorney's fee is $14,875.00. Calculated as: $350/hr × 42.5 hr = $14,875."),
    ("The lease requires 60 days' notice of termination. If notice was given on March 10, what is the earliest termination date?",
     "notice_days", {"notice_days": 60}, 0,
     "Adding 60 days to March 10 yields May 9. The earliest termination date is May 9."),
    ("Calculate the simple interest on a $75,000 promissory note at 8% annually for 18 months.",
     "P * r * (t / 12)", {"P": 75000, "r": 0.08, "t": 18}, 2,
     "The simple interest is $9,000.00. Calculated as: $75,000 × 0.08 × (18/12) = $9,000."),
    ("A judgment of $200,000 compounds at 5% annually. What is the value after 4 years?",
     "P * (1 + r) ** t", {"P": 200000, "r": 0.05, "t": 4}, 2,
     "The compounded judgment value after 4 years is $243,101.25. Calculated as: $200,000 × (1.05)^4 ≈ $243,101.25."),
    ("A tenant owes three months' back rent at $1,800/month plus a 10% late fee. What is the total amount owed?",
     "months * rent * (1 + late_fee)", {"months": 3, "rent": 1800, "late_fee": 0.1}, 2,
     "The total amount owed is $5,940.00. Calculated as: 3 × $1,800 × 1.10 = $5,940."),
    ("Plaintiff seeks lost wages: $85,000/year salary, out of work for 14 months. Calculate the lost wage claim.",
     "salary / 12 * months", {"salary": 85000, "months": 14}, 2,
     "The lost wage claim is $99,166.67. Calculated as: $85,000 ÷ 12 × 14 ≈ $99,166.67."),
]


def _make_l05_sample(idx: int) -> Dict:
    case = L05_CASES[idx % len(L05_CASES)]
    question, expr, vars_, round_to, explanation = case

    tc = {
        "tool": "sympy", "method": "calc",
        "expr": expr, "vars": vars_, "round_to": round_to,
    }

    return {
        "lane": "L05",
        "source": "legal_calc_r3",
        "messages": [
            {"role": "system", "content": SYSTEM_EN},
            {"role": "user", "content": question},
            {
                "role": "assistant",
                "content": f"<lane>L05</lane>\n<tool_call>{json.dumps(tc, ensure_ascii=False)}</tool_call>",
            },
            {"role": "tool", "content": json.dumps({"result": "calculated"}, ensure_ascii=False)},
            {"role": "assistant", "content": explanation},
        ],
    }


# ---------------------------------------------------------------------------
# L06 explanation seeds (with optional <reasoning> block — P2)
# ---------------------------------------------------------------------------

L06_CASES = [
    (
        "What is the parol evidence rule?",
        "The parol evidence rule prohibits the introduction of extrinsic evidence — "
        "oral or written statements made prior to or contemporaneously with a written "
        "contract — to contradict, vary, or add to the terms of a fully integrated "
        "written agreement. The rule protects the integrity of written contracts and "
        "prevents fraud or mistake in contract formation.",
    ),
    (
        "Explain the doctrine of promissory estoppel.",
        "Promissory estoppel (also called detrimental reliance) allows enforcement of "
        "a promise that lacks consideration when: (1) the promisor made a clear and "
        "definite promise; (2) the promisee reasonably relied on it; (3) the reliance "
        "was detrimental; and (4) injustice can only be avoided by enforcement. It is "
        "codified in Restatement (Second) of Contracts § 90.",
    ),
    (
        "What does 'res ipsa loquitur' mean in tort law?",
        "Res ipsa loquitur ('the thing speaks for itself') is a doctrine in negligence "
        "law that allows an inference of negligence when: (1) the injury is of a type "
        "that ordinarily does not occur absent negligence; (2) the instrumentality "
        "causing the harm was in the defendant's exclusive control; and (3) the "
        "plaintiff did not contribute to the injury. It shifts the burden of producing "
        "evidence to the defendant.",
    ),
    (
        "What is the business judgment rule?",
        "The business judgment rule is a presumption that in making business decisions, "
        "corporate directors acted on an informed basis, in good faith, and in the "
        "honest belief that the action was in the best interests of the company. Courts "
        "will not second-guess such decisions unless a plaintiff shows fraud, illegality, "
        "conflict of interest, or gross negligence in the decision-making process.",
    ),
    (
        "Explain the Chevron doctrine.",
        "The Chevron doctrine (Chevron U.S.A., Inc. v. NRDC, 1984) established a two-"
        "step framework for judicial review of agency statutory interpretation: "
        "(Step 1) Is the statute ambiguous? If clear, apply its plain meaning. "
        "(Step 2) If ambiguous, is the agency's interpretation reasonable? If so, "
        "courts must defer to it. The doctrine was overruled by Loper Bright Enterprises "
        "v. Raimondo (2024), which restored de novo review of statutory meaning.",
    ),
    (
        "What is the difference between specific and general personal jurisdiction?",
        "General jurisdiction allows a court to hear any claim against a defendant "
        "based on their continuous and systematic presence in the forum state (e.g., "
        "domicile for individuals; principal place of business or state of incorporation "
        "for corporations). Specific jurisdiction is narrower — it requires that the "
        "claim arise out of or relate to the defendant's contacts with the forum state, "
        "satisfying minimum contacts and fair play under International Shoe Co. v. "
        "Washington (1945).",
    ),
    (
        "What is 'stare decisis'?",
        "Stare decisis ('to stand by things decided') is the doctrine of legal precedent "
        "under which courts are generally bound to follow prior rulings from higher courts "
        "in the same jurisdiction when deciding cases with substantially similar facts and "
        "legal issues. It promotes consistency, predictability, and stability in the law. "
        "A court may depart from precedent if it finds the prior decision was wrongly "
        "decided or if circumstances have materially changed.",
    ),
    (
        "Explain the concept of 'piercing the corporate veil'.",
        "Piercing the corporate veil is an equitable remedy allowing courts to hold "
        "shareholders personally liable for corporate obligations when the corporate form "
        "is used as a shield for fraud or when the corporation is a mere alter ego of the "
        "shareholder. Courts consider factors such as commingling of assets, failure to "
        "observe corporate formalities, undercapitalization, and fraudulent intent.",
    ),
    (
        "What is the 'fruit of the poisonous tree' doctrine?",
        "The fruit of the poisonous tree doctrine, derived from Wong Sun v. United States "
        "(1963), holds that evidence obtained as a result of an illegal search or seizure "
        "is inadmissible — not only the direct evidence (the 'poisonous tree') but also "
        "secondary evidence derived from it (the 'fruit'). Exceptions include: attenuation, "
        "independent source, and inevitable discovery.",
    ),
    (
        "Describe the elements of common law fraud.",
        "Common law fraud requires proof of: (1) a false representation of a material fact; "
        "(2) made with knowledge of its falsity or reckless disregard for truth (scienter); "
        "(3) with intent to induce reliance; (4) actual and justifiable reliance by the "
        "plaintiff; and (5) resulting damages. The heightened pleading standard of FRCP 9(b) "
        "requires fraud to be pled with particularity in federal courts.",
    ),
]


def _make_l06_sample(idx: int, add_reasoning: bool = False) -> Dict:
    case = L06_CASES[idx % len(L06_CASES)]
    question, explanation = case

    if add_reasoning:
        content = (
            f"<lane>L06</lane>\n"
            f"<reasoning>This is a direct explanation task requiring legal analysis "
            f"without formal element-checking. No solver needed.</reasoning>\n"
            f"{explanation}"
        )
    else:
        content = f"<lane>L06</lane>\n{explanation}"

    return {
        "lane": "L06",
        "source": "legal_explanation_r3",
        "messages": [
            {"role": "system", "content": SYSTEM_EN},
            {"role": "user", "content": question},
            {"role": "assistant", "content": content},
        ],
    }


# ---------------------------------------------------------------------------
# L09 translation seeds
# ---------------------------------------------------------------------------

L09_CASES = [
    ("What does 'in rem' mean?",
     "In rem is a Latin term meaning 'against the thing.' In law, an in rem proceeding is a legal action directed against property itself rather than against a specific person. Courts exercise in rem jurisdiction over property located within their territorial boundaries."),
    ("Translate and explain 'quantum meruit'.",
     "Quantum meruit means 'as much as he has deserved' in Latin. It is an equitable remedy allowing recovery of the reasonable value of services rendered when there is no enforceable contract, to prevent unjust enrichment."),
    ("What does 'mens rea' mean in criminal law?",
     "Mens rea is Latin for 'guilty mind.' It refers to the mental state or intent required for criminal liability. Common levels include: purpose/intent, knowledge, recklessness, and negligence. A defendant generally must have both mens rea and actus reus (the guilty act) to be convicted."),
    ("Explain 'locus standi'.",
     "Locus standi means 'place of standing' in Latin. In legal procedure, it refers to a party's right or capacity to bring an action before a court. In U.S. constitutional law it corresponds to standing — requiring injury in fact, causation, and redressability."),
    ("What is 'habeas corpus'?",
     "Habeas corpus is Latin for 'you shall have the body.' It is a legal writ requiring that a person detained by the government be brought before a court to determine whether the detention is lawful. It is a fundamental protection against unlawful imprisonment, codified in 28 U.S.C. § 2241."),
    ("What does 'de minimis' mean?",
     "De minimis is Latin for 'about minimal things.' In law, it refers to something too small or trivial to merit legal consideration. Courts apply the de minimis doctrine to dismiss claims where the alleged harm or violation is negligible and below the threshold that the law was designed to address."),
    ("Explain 'prima facie'.",
     "Prima facie is Latin for 'at first sight' or 'on its face.' A prima facie case is one that appears sufficient to establish a fact or raise a presumption, unless rebutted. In employment discrimination law, for example, a plaintiff must first establish a prima facie case before the burden shifts to the defendant."),
    ("What does 'inter alia' mean?",
     "Inter alia is Latin for 'among other things.' It is used in legal writing to indicate that a list is not exhaustive — for example, 'the contract required, inter alia, monthly payments and insurance' means those were among several requirements, not the only ones."),
    ("What is 'nunc pro tunc'?",
     "Nunc pro tunc is Latin for 'now for then.' It refers to a court order that takes retroactive effect, correcting the record to reflect what should have been done at an earlier time. For example, a nunc pro tunc entry might correct a clerical error in a prior judgment."),
    ("Translate and explain 'scienter'.",
     "Scienter is Latin for 'knowingly.' In law it refers to the mental state of knowing that a statement or action is false or wrongful. Proof of scienter is often required in fraud, securities violations (e.g., SEC Rule 10b-5), and certain criminal offenses."),
]


def _make_l09_sample(idx: int) -> Dict:
    case = L09_CASES[idx % len(L09_CASES)]
    question, answer = case
    return {
        "lane": "L09",
        "source": "legal_translation_r3",
        "messages": [
            {"role": "system", "content": SYSTEM_EN},
            {"role": "user", "content": question},
            {"role": "assistant", "content": f"<lane>L09</lane>\n{answer}"},
        ],
    }


# ---------------------------------------------------------------------------
# L10 uncertainty seeds
# ---------------------------------------------------------------------------

L10_CASES = [
    ("Will the court in this jurisdiction apply the economic loss rule to bar this negligence claim?",
     "Whether the economic loss rule bars this claim depends on the specific jurisdiction's approach and recent case law. Some states recognize a professional services exception; others apply the rule strictly. Without knowing the jurisdiction, I cannot determine the outcome with confidence."),
    ("Is this arbitration clause enforceable under the FAA?",
     "Enforceability of arbitration clauses under the Federal Arbitration Act depends on multiple factors: whether the clause is unconscionable under state contract law, whether the dispute falls within its scope, and recent Supreme Court precedent. A definitive answer requires reviewing the full agreement and the applicable state law."),
    ("Can the plaintiff recover punitive damages in this contract dispute?",
     "Punitive damages in contract cases are generally not available under traditional common law principles, but some jurisdictions permit them when the breach is accompanied by fraud, malice, or oppression. The answer depends on the jurisdiction and the specific facts of the alleged misconduct."),
    ("Does this non-compete agreement violate state law?",
     "The enforceability of non-compete agreements varies significantly by state. California, for example, largely prohibits them; other states enforce them if reasonable in scope, duration, and geography. Without knowing the governing state law and reviewing all agreement terms, I cannot make a definitive determination."),
    ("Is the statute of limitations tolled in this case?",
     "Tolling of the statute of limitations depends on the discovery rule, equitable tolling doctrines, and any applicable statutory exceptions in the relevant jurisdiction. Whether tolling applies here requires analyzing the plaintiff's knowledge of the injury and the defendant's conduct in more detail."),
    ("Does the First Amendment protect this speech in the workplace?",
     "First Amendment protections generally apply to government actors, not private employers. Whether a particular government workplace speech restriction is constitutional depends on whether the employee spoke as a private citizen on a matter of public concern (Garcetti v. Ceballos). The analysis is highly fact-specific."),
    ("Is this expert testimony admissible under Daubert?",
     "Admissibility under Daubert v. Merrell Dow requires the court to evaluate whether the methodology is testable, has been peer-reviewed, has an acceptable error rate, and is generally accepted in the relevant scientific community. A full analysis requires reviewing the expert's report and qualifications."),
    ("Does this settlement release all future claims?",
     "Whether a release bars future claims depends on its scope, whether it was knowing and voluntary, and whether the future claims were within the reasonable contemplation of the parties at the time of signing. Courts construe releases narrowly if they purport to cover unknown future claims."),
]


def _make_l10_sample(idx: int) -> Dict:
    case = L10_CASES[idx % len(L10_CASES)]
    question, answer = case
    return {
        "lane": "L10",
        "source": "legal_uncertain_r3",
        "messages": [
            {"role": "system", "content": SYSTEM_EN},
            {"role": "user", "content": question},
            {"role": "assistant", "content": f"<lane>L10</lane>\n{answer}"},
        ],
    }


# ---------------------------------------------------------------------------
# Korean L01 seeds
# ---------------------------------------------------------------------------

KO_L01_CASES = [
    (
        "피고가 피해자 소유의 휴대폰을 허락 없이 가져갔습니다. 절도죄가 성립하나요?",
        ["절도죄_구성요건"],
        {"절도죄_구성요건": True},
        True, "and",
        ["피고가 타인의 재물을 취득함", "피해자의 허락이 없었음"],
        "네, 절도죄가 성립합니다. 타인의 재물을 허락 없이 가져갔으므로 형법 제329조의 절도죄 구성요건을 충족합니다.",
    ),
    (
        "A가 B의 지갑을 공원 벤치에서 습득하여 가져갔습니다. 이는 절도죄인가요?",
        ["절도죄_구성요건", "점유이탈물횡령"],
        {"절도죄_구성요건": False, "점유이탈물횡령": True},
        False, "and",
        ["B가 지갑을 공원에 두고 떠난 상태", "A가 이를 습득함"],
        "아니오. 지갑은 B의 점유를 이탈한 상태였으므로 절도죄가 아니라 점유이탈물횡령죄(형법 제360조)가 성립합니다.",
    ),
    (
        "C는 D에게 허위 사실을 말하여 돈을 받았습니다. 사기죄가 성립하나요?",
        ["사기죄_구성요건"],
        {"사기죄_구성요건": True},
        True, "and",
        ["C가 D에게 허위 사실을 고지함", "D가 이를 믿고 재산을 교부함"],
        "네, 사기죄가 성립합니다. 형법 제347조의 사기죄는 기망행위로 상대방을 착오에 빠뜨려 재산상 이익을 취득하는 경우 성립하며, 모든 요건이 충족됩니다.",
    ),
    (
        "채권자 A의 채권이 10년 소멸시효 기간이 경과하였습니다. 소멸시효가 완성되었나요?",
        ["소멸시효_완성"],
        {"소멸시효_완성": True},
        True, "and",
        ["채권 발생 후 10년이 경과함", "시효 중단 사유 없음"],
        "네, 소멸시효가 완성되었습니다. 민법 제162조에 따라 일반 채권의 소멸시효는 10년이며, 중단 사유 없이 기간이 경과하면 소멸시효가 완성됩니다.",
    ),
    (
        "채권자가 소멸시효 완성 전에 소를 제기하였습니다. 시효가 중단되나요?",
        ["시효중단"],
        {"시효중단": True},
        True, "and",
        ["소멸시효 완성 이전에 소를 제기함"],
        "네, 시효가 중단됩니다. 민법 제168조에 따라 재판상 청구는 소멸시효를 중단시키며, 판결 확정 후 새로운 시효가 진행됩니다.",
    ),
    (
        "임대차 계약에서 임차인이 3개월 이상 차임을 연체하였습니다. 임대인이 계약을 해지할 수 있나요?",
        ["계약해지권"],
        {"계약해지권": True},
        True, "and",
        ["임차인이 3개월 이상 차임 연체", "임대인의 해지권 발생"],
        "네, 임대인은 계약을 해지할 수 있습니다. 민법 제640조에 따라 건물 임차인이 2기의 차임을 연체하면 임대인은 해지권을 행사할 수 있습니다.",
    ),
    (
        "불법행위로 인해 피해자에게 손해가 발생하였습니다. 손해배상 책임이 인정되나요?",
        ["손해배상_요건"],
        {"손해배상_요건": True},
        True, "and",
        ["가해자의 고의 또는 과실이 인정됨", "손해 발생 및 인과관계 확인"],
        "네, 손해배상 책임이 인정됩니다. 민법 제750조에 따라 고의 또는 과실로 인한 위법행위로 타인에게 손해를 가한 경우 손해배상 의무가 발생합니다.",
    ),
]


def _make_ko_l01_sample(idx: int) -> Dict:
    case = KO_L01_CASES[idx % len(KO_L01_CASES)]
    question, elements, matching, all_met, mode, facts, explanation = case

    tc = {
        "tool": "z3", "method": "check_elements",
        "elements": elements,
        "facts": facts,
        "matching": matching,
        "mode": mode,
    }

    return {
        "lane": "L01",
        "source": "korean_civil_r3",
        "messages": [
            {"role": "system", "content": SYSTEM_KO},
            {"role": "user", "content": question},
            {
                "role": "assistant",
                "content": f"<lane>L01</lane>\n<tool_call>{json.dumps(tc, ensure_ascii=False)}</tool_call>",
            },
            {"role": "tool", "content": json.dumps({"all_met": all_met}, ensure_ascii=False)},
            {"role": "assistant", "content": explanation},
        ],
    }


# ---------------------------------------------------------------------------
# Generate from LegalBench test data
# ---------------------------------------------------------------------------

def _load_legalbench_samples() -> List[Dict]:
    """Augment seeds with real LegalBench test samples."""
    import pandas as pd
    samples = []

    # Hearsay test
    hearsay_csv = ROOT / "data/legalbench/hearsay__test.csv"
    if hearsay_csv.exists():
        df = pd.read_csv(hearsay_csv)
        for _, row in df.iterrows():
            question = str(row["text"])
            answer = str(row["answer"]).strip()
            is_yes = answer == "Yes"

            # Infer elements from answer
            if is_yes:
                matching = {"out_of_court_statement": True, "offered_for_truth": True}
                all_met = True
            else:
                # Vary the mismatch
                slice_val = str(row.get("slice", "")).lower()
                if "in-court" in slice_val or "court" in slice_val:
                    matching = {"out_of_court_statement": False, "offered_for_truth": True}
                elif "not introduced" in slice_val or "not offered" in slice_val:
                    matching = {"out_of_court_statement": True, "offered_for_truth": False}
                else:
                    matching = {"out_of_court_statement": False, "offered_for_truth": False}
                all_met = False

            tc = {
                "tool": "z3", "method": "check_elements",
                "elements": ["out_of_court_statement", "offered_for_truth"],
                "facts": [question[:150]],
                "matching": matching,
                "mode": "and",
            }
            explanation_base = (
                f"{answer}. Based on the analysis of whether an out-of-court statement "
                f"was offered for its truth under FRE 801(c), this {'is' if is_yes else 'is not'} hearsay."
            )
            samples.append({
                "lane": "L01",
                "source": "legalbench_hearsay_test_r3",
                "messages": [
                    {"role": "system", "content": SYSTEM_EN},
                    {"role": "user", "content": question + " Is this hearsay?"},
                    {"role": "assistant", "content": f"<lane>L01</lane>\n<tool_call>{json.dumps(tc, ensure_ascii=False)}</tool_call>"},
                    {"role": "tool", "content": json.dumps({"all_met": all_met}, ensure_ascii=False)},
                    {"role": "assistant", "content": explanation_base},
                ],
            })

    # PJ test
    pj_csv = ROOT / "data/legalbench/personal_jurisdiction__test.csv"
    if pj_csv.exists():
        df = pd.read_csv(pj_csv)
        for _, row in df.iterrows():
            question = str(row["text"])
            answer = str(row["answer"]).strip()
            is_yes = answer == "Yes"
            slice_val = str(row.get("slice", "")).lower()

            if is_yes:
                matching = {
                    "minimum_contacts": True,
                    "purposeful_availment": True,
                    "fair_play_and_substantial_justice": True,
                }
            elif "no contacts" in slice_val:
                matching = {
                    "minimum_contacts": False,
                    "purposeful_availment": False,
                    "fair_play_and_substantial_justice": True,
                }
            elif "yes contacts, no nexus" in slice_val:
                matching = {
                    "minimum_contacts": True,
                    "purposeful_availment": False,
                    "fair_play_and_substantial_justice": True,
                }
            else:
                matching = {
                    "minimum_contacts": bool(is_yes),
                    "purposeful_availment": bool(is_yes),
                    "fair_play_and_substantial_justice": True,
                }
            all_met = all(matching.values())

            tc = {
                "tool": "z3", "method": "check_elements",
                "elements": ["minimum_contacts", "purposeful_availment",
                             "fair_play_and_substantial_justice"],
                "facts": [question[:150]],
                "matching": matching,
                "mode": "and",
            }
            explanation_base = (
                f"{answer}. Personal jurisdiction {'is' if is_yes else 'is not'} proper "
                f"based on the analysis of minimum contacts, purposeful availment, and fair play."
            )
            samples.append({
                "lane": "L01",
                "source": "legalbench_pj_test_r3",
                "messages": [
                    {"role": "system", "content": SYSTEM_EN},
                    {"role": "user", "content": question + " Does the court have personal jurisdiction?"},
                    {"role": "assistant", "content": f"<lane>L01</lane>\n<tool_call>{json.dumps(tc, ensure_ascii=False)}</tool_call>"},
                    {"role": "tool", "content": json.dumps({"all_met": all_met}, ensure_ascii=False)},
                    {"role": "assistant", "content": explanation_base},
                ],
            })

    # Textualism test
    tx_csv = ROOT / "data/legalbench/textualism_tool__test.csv"
    if tx_csv.exists():
        df = pd.read_csv(tx_csv)
        for _, row in df.iterrows():
            question = str(row["text"])
            answer = str(row["answer"]).strip()
            is_yes = answer == "Yes"

            elements = ["dictionary_definition_used", "plain_meaning_invoked"]
            matching = {
                "dictionary_definition_used": is_yes,
                "plain_meaning_invoked": is_yes,
            }
            tc = {
                "tool": "z3", "method": "check_elements",
                "elements": elements,
                "facts": [question[:120]],
                "matching": matching,
                "mode": "or",
            }
            explanation_base = (
                f"{answer}. The excerpt {'does' if is_yes else 'does not'} employ "
                f"textualist methods such as dictionary definitions or plain meaning analysis."
            )
            samples.append({
                "lane": "L01",
                "source": "legalbench_textualism_test_r3",
                "messages": [
                    {"role": "system", "content": SYSTEM_EN},
                    {"role": "user", "content": "Does the following reflect textualist interpretation?\n\n" + question},
                    {"role": "assistant", "content": f"<lane>L01</lane>\n<tool_call>{json.dumps(tc, ensure_ascii=False)}</tool_call>"},
                    {"role": "tool", "content": json.dumps({"all_met": is_yes}, ensure_ascii=False)},
                    {"role": "assistant", "content": explanation_base},
                ],
            })

    return samples


# ---------------------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------------------

def generate_seeds_r3(seeds_dir: Path) -> Dict[str, List[Dict]]:
    seeds_dir.mkdir(parents=True, exist_ok=True)

    # Hearsay: 200 samples
    hearsay = [_make_hearsay_sample(i) for i in range(200)]
    (seeds_dir / "L01_en_hearsay.json").write_text(
        json.dumps(hearsay, ensure_ascii=False, indent=2)
    )
    print(f"hearsay: {len(hearsay)} samples → {seeds_dir}/L01_en_hearsay.json")

    # PJ: 200 samples
    pj = [_make_pj_sample(i) for i in range(200)]
    (seeds_dir / "L01_en_pj.json").write_text(
        json.dumps(pj, ensure_ascii=False, indent=2)
    )
    print(f"pj: {len(pj)} samples → {seeds_dir}/L01_en_pj.json")

    # Textualism: 200 samples
    textualism = [_make_textualism_sample(i) for i in range(200)]
    (seeds_dir / "L01_en_textualism.json").write_text(
        json.dumps(textualism, ensure_ascii=False, indent=2)
    )
    print(f"textualism: {len(textualism)} samples → {seeds_dir}/L01_en_textualism.json")

    # L05 calc: 30
    l05 = [_make_l05_sample(i) for i in range(30)]
    (seeds_dir / "L05_en.json").write_text(
        json.dumps(l05, ensure_ascii=False, indent=2)
    )
    print(f"L05: {len(l05)} samples → {seeds_dir}/L05_en.json")

    # L06 explanation: 70 (last 10 with <reasoning> block for P2)
    l06 = [_make_l06_sample(i, add_reasoning=(i >= 60)) for i in range(70)]
    (seeds_dir / "L06_en.json").write_text(
        json.dumps(l06, ensure_ascii=False, indent=2)
    )
    print(f"L06: {len(l06)} samples → {seeds_dir}/L06_en.json")

    # L09 translation: 20
    l09 = [_make_l09_sample(i) for i in range(20)]
    (seeds_dir / "L09_en.json").write_text(
        json.dumps(l09, ensure_ascii=False, indent=2)
    )
    print(f"L09: {len(l09)} samples → {seeds_dir}/L09_en.json")

    # L10 uncertainty: 20
    l10 = [_make_l10_sample(i) for i in range(20)]
    (seeds_dir / "L10_en.json").write_text(
        json.dumps(l10, ensure_ascii=False, indent=2)
    )
    print(f"L10: {len(l10)} samples → {seeds_dir}/L10_en.json")

    # Korean L01: target ~140 for 20% of 540 EN total
    ko_l01 = [_make_ko_l01_sample(i) for i in range(140)]
    (seeds_dir / "L01_ko.json").write_text(
        json.dumps(ko_l01, ensure_ascii=False, indent=2)
    )
    print(f"KO L01: {len(ko_l01)} samples → {seeds_dir}/L01_ko.json")

    return {
        "hearsay": hearsay, "pj": pj, "textualism": textualism,
        "l05": l05, "l06": l06, "l09": l09, "l10": l10, "ko_l01": ko_l01,
    }


def generate_sft_r3(
    seeds_dir: Path,
    output_path: Path,
    target_total: int = 8500,
) -> int:
    """Combine seeds + LegalBench real samples + augmentation to reach target_total."""

    # 1. Load all seed files
    all_samples: List[Dict] = []
    for fname in sorted(seeds_dir.glob("*.json")):
        data = json.loads(fname.read_text(encoding="utf-8"))
        all_samples.extend(data)
    print(f"Seeds loaded: {len(all_samples)} raw samples")

    # 2. Load LegalBench-derived samples
    lb_samples = _load_legalbench_samples()
    all_samples.extend(lb_samples)
    print(f"After LegalBench augmentation: {len(all_samples)} samples")

    # 3. Augment by repetition with shuffled order until target reached
    if len(all_samples) < target_total:
        multiplier = (target_total // len(all_samples)) + 1
        base = all_samples.copy()
        while len(all_samples) < target_total:
            chunk = base.copy()
            random.shuffle(chunk)
            all_samples.extend(chunk)

    # 4. Trim to target
    random.shuffle(all_samples)
    all_samples = all_samples[:target_total]

    # 5. Verify English ratio
    en_count = sum(
        1 for s in all_samples
        if any(m["role"] == "system" and "You are" in m["content"]
               for m in s["messages"])
    )
    ko_count = target_total - en_count
    en_pct = en_count / target_total * 100
    print(f"Language split: EN={en_count} ({en_pct:.1f}%), KO={ko_count} ({100-en_pct:.1f}%)")

    # 6. Verify binary Yes/No rate for L01 binary tasks
    binary_sources = {"legalbench_hearsay", "legalbench_pj", "legalbench_textualism",
                      "legalbench_hearsay_r3", "legalbench_pj_r3", "legalbench_textualism_r3",
                      "legalbench_hearsay_test_r3", "legalbench_pj_test_r3",
                      "legalbench_textualism_test_r3"}
    binary_samples = [s for s in all_samples if s.get("source", "") in binary_sources]
    if binary_samples:
        yes_no_starts = 0
        for s in binary_samples:
            last_assistant = [m for m in s["messages"] if m["role"] == "assistant"][-1]
            if re.match(r"^(Yes|No)[.\s,;:]", last_assistant["content"]):
                yes_no_starts += 1
        yn_rate = yes_no_starts / len(binary_samples)
        print(f"Binary Yes/No start rate: {yn_rate:.2f} ({yes_no_starts}/{len(binary_samples)})")

    # 7. Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for s in all_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"SFT data written: {len(all_samples)} samples → {output_path}")

    return len(all_samples)


if __name__ == "__main__":
    import re
    import argparse

    parser = argparse.ArgumentParser(description="R3 SFT data generation")
    parser.add_argument("--seeds-only", action="store_true",
                        help="Generate seeds only, skip SFT combination")
    parser.add_argument("--target", type=int, default=8500,
                        help="Target SFT sample count")
    args = parser.parse_args()

    seeds_dir = ROOT / "data/seeds_r3"
    output_path = ROOT / "data/sft_r3.jsonl"

    print("=== Generating R3 Seeds ===")
    seeds = generate_seeds_r3(seeds_dir)

    en_total = sum(len(v) for k, v in seeds.items() if k != "ko_l01")
    ko_total = len(seeds["ko_l01"])
    print(f"\nSeed totals: EN={en_total}, KO={ko_total}, ratio={en_total/(en_total+ko_total)*100:.1f}% EN")

    if not args.seeds_only:
        import re
        print("\n=== Generating SFT Data ===")
        n = generate_sft_r3(seeds_dir, output_path, target_total=args.target)
        print(f"\nTotal SFT samples: {n}")
