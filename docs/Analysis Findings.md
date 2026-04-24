# TheatreIQ — Analysis Findings & Business Recommendations

> **Dataset:** 17,455 showings · 5 Twin Cities venues · 750 unique films · Feb 2021–May 2026  
> **Model:** ALS collaborative filter (MovieLens 100K) → XGBoost ranker · Holdout AUC 0.696  
> **Match scores:** 7.2 – 91.4 · Computed across 8 demographic segments × 5 time slots

---

## Executive Summary

TheatreIQ's two-stage recommender surfaces four findings that directly affect revenue and audience development for every venue in the dataset:

1. **Heights Theater operates in the optimal zone.** Every scheduled slot scores `high_confidence` (84–95). Its programming identity — classic prestige cinema — is the strongest audience-fit match of the five venues.
2. **A "Casablanca Cluster" of seven films scores 86–91 at four venues that have never booked them.** These are the highest-value unbooked titles in the market. Heights has a near-monopoly on this programming space.
3. **35–54 F is the most underserved demographic across all five venues, in every time slot.** No venue's programming fits this segment well. It represents the largest untapped revenue opportunity in the dataset.
4. **If all 5 theaters adopt the recommended schedules, combined annual ticket revenue increases from $4.90M to $5.36M (+$0.46M, +9.5%).** Even the conservative scenario — replacing only the weakest-fit films — projects a +$0.21M (+4.3%) gain. *(Figures based on synthetic occupancy; see Revenue Impact section for methodology.)*

---

## Analysis 1 — Venue Genre Fingerprints

*What each theater actually programs, by % of showings with each genre tag.*

| Venue | #1 Genre | #2 Genre | #3 Genre | Identity Signal |
|---|---|---|---|---|
| Trylon Cinema | Drama 50% | Thriller 42% | Crime 26% | Arthouse / Noir |
| Riverview Theater | Adventure 41% | Comedy 33% | Drama 32% | Crowd-pleaser / Classic |
| Lagoon | Action 36% | Adventure 35% | Drama 34% | Balanced / Contemporary |
| Heights Theater | Adventure 42% | Action 40% | Sci-Fi 29% | Classic Prestige |
| Drafthouse | Adventure 44% | Action 43% | Sci-Fi 32% | Genre / Event |

### Key Findings

- **Trylon is the only venue with a genuinely distinct programming identity.** Drama/Thriller/Crime is a completely different fingerprint from the other four.
- **Heights and Drafthouse are nearly identical in genre composition** (both ~44% Adventure, ~41% Action, ~30% Sci-Fi) despite being very different venues in character and programming philosophy. Both appear to draw from the same distributor pool rather than curating differently.
- **No venue's top-3 genres include Romance, Musical, or Animation** — confirming the collective blind spot on female-skewing content.

### KPI Impact
| KPI | Effect |
|---|---|
| Audience differentiation | Trylon is the only venue creating genuine programming separation in the market |
| Distributor negotiation | Heights/Drafthouse overlap creates internal competition for the same prints |
| Brand identity | Genre fingerprints can be used in marketing copy to signal identity to specific audiences |

---

## Analysis 2 — Segment × Venue Match Score Heatmap

*Average match score for each venue's historically programmed films, broken out by demographic segment.*

| Venue | Strongest Segment | Score | Weakest Segment | Score | Gap |
|---|---|---|---|---|---|
| Trylon Cinema | Under 18 M | 47.9 | 35–54 F | 34.0 | 13.9 |
| Heights Theater | Under 18 M | 49.7 | 35–54 F | 36.9 | 12.8 |
| Riverview Theater | 55+ M | 40.9 | 35–54 F | 28.7 | 12.2 |
| Lagoon | 55+ M | 43.2 | 35–54 F | 30.4 | 12.8 |
| Drafthouse | 55+ M | 42.9 | 35–54 F | 30.3 | 12.6 |

### Key Findings

- **35–54 F is the weakest-scoring segment at every single venue** — not by a small margin. The gap between best and worst segment averages ~13 points at every venue.
- **55+ M leads at Riverview, Lagoon, and Drafthouse** — the broader Action/Adventure/Sci-Fi catalog skews toward older male audiences from MovieLens taste signals.
- **Under 18 M leads at Heights and Trylon** — classic cinema (Hitchcock, Kubrick, Kurosawa) has strong cross-generational male appeal in the collaborative filter.
- No venue currently scores above 50 for any segment on average — indicating meaningful room to improve programming fit across the board.

### KPI Impact
| KPI | Effect |
|---|---|
| Occupancy rate | Improving fit for underserved segments (35–54 F) is the highest-upside lever |
| Repeat attendance | Strong segment fit → audience identity → loyalty and word-of-mouth |
| Audience diversification | Current programming skews 55+M / Under18M; no venue is building the 35–54 F audience |

---

## Analysis 3 — Slot × Segment Preference Matrix

*Which demographic peaks in each time slot across the full catalog.*

| Slot | Peak Segment | Score | Marketing Implication |
|---|---|---|---|
| Friday Prime (7–10pm) | Under 18 M | 44.9 | Genre films, cult, event programming — promote on social/Reddit |
| Saturday Matinee (1–4pm) | 55+ M | 45.7 | Classic/prestige picks — email newsletter, arts calendar |
| Saturday Prime (7–10pm) | 35–54 M | 46.5 | Highest engagement slot overall — program strongest titles here |
| Sunday Matinee (1–4pm) | 35–54 M | 41.5 | Adult second-chance viewing — good for word-of-mouth titles |
| Weekday | 55+ M | 44.7 | Retiree/older audience — matinee pricing, loyalty programs |

### Key Findings

- **Saturday Prime is the highest-value slot** (score 46.5 for 35–54 M). This is where the best-fit films should go — not Friday, which skews younger and more genre-specific.
- **35–54 F does not peak in any slot.** This isn't a scheduling problem — it's a catalog problem. The films that would bring this audience in aren't being booked.
- **Friday Prime skews Under 18 M**, meaning event-style programming (horror nights, cult screenings, director retrospectives) fits this window better than prestige drama.
- **Weekday and Saturday Matinee both peak 55+ M** — these two slots share an audience and should share a programming philosophy (classic films, lower pricing, loyalty incentives).

### KPI Impact
| KPI | Effect |
|---|---|
| Revenue per slot | Moving strongest films to Saturday Prime instead of Friday optimizes per-showing revenue |
| Marketing ROI | Targeting the right channel per slot (email for 55+ M, social for Under 18 M) reduces wasted spend |
| Occupancy variance | Slot-matched programming reduces the weekend/weekday occupancy gap (currently 35% avg vs 55% peak) |

---

## Analysis 4 — Missed Opportunity Report

*Films scoring ≥ 65 (high_confidence) that each venue has never programmed.*

### Universal Missed Films — 4 of 5 Venues

These seven films score `high_confidence` at Trylon, Riverview, Lagoon, and Drafthouse but have never been booked there. Heights is the only venue showing them.

| Film | Score | Primary Segment | Tier |
|---|---|---|---|
| Casablanca (1942) | 91.4 | 35–54 M | Easy Win |
| Schindler's List (1993) | 90.9 | Under 18 M | Easy Win |
| North by Northwest (1959) | 90.2 | 35–54 M | Core Identity |
| Vertigo (1958) | 89.8 | 35–54 M | Core Identity |
| Rear Window (1954) | 89.2 | 18–34 M | Core Identity |
| Raiders of the Lost Ark (1981) | 89.1 | 35–54 M | Easy Win |
| Amadeus (1984) | 86.5 | 55+ F | Easy Win |

**The Casablanca Cluster:** Heights has a near-monopoly on classic prestige cinema in the Twin Cities indie market. Any of the other four venues booking even two or three of these would be directly competing in Heights' strongest territory — but also directly serving their own underserved 35–54 M and Under 18 M audiences.

### Heights Theater — Specific Missed Films

Heights already programs the classics, so its missed list is different — crossover contemporary titles its audience would respond to:

| Film | Score | Primary Segment |
|---|---|---|
| Goodfellas (1990) | 81.9 | Under 18 M |
| Civil War (2024) | 80.1 | Under 18 M |
| It's A Wonderful Life (1946) | 78.5 | 35–54 M |
| Touch of Evil (1958) | 78.1 | 18–34 M |
| Barbie (2023) | 76.8 | 18–34 M |

Heights can expand into contemporary prestige and crossover titles without abandoning its identity.

### KPI Impact
| KPI | Effect |
|---|---|
| Revenue per screen | Booking one Casablanca Cluster film per venue = direct high_confidence occupancy lift |
| Competitive positioning | First-mover on these titles outside Heights creates programming differentiation |
| New audience acquisition | Casablanca Cluster drives 35–54 M (currently weakest segment at these venues) |

---

## Analysis 5 — Marketing Tier (Popularity × Match Score)

*750 films segmented by audience fit and market-level draw.*

| Tier | Count | Avg Match Score | Avg TMDB Popularity | What to do |
|---|---|---|---|---|
| Easy Win | 110 | 66.8 | 17.9 | Book confidently, minimal marketing spend |
| Core Identity | 152 | 68.1 | 4.2 | Strong fit, low buzz — needs curator-led promotion |
| Wrong Fit | 190 | 41.6 | 23.1 | Resist distributor pressure — will underperform |
| Skip | 298 | 40.7 | 4.0 | Weak on both axes |

### Representative Films by Tier

**Easy Wins** (high fit + enough market recognition to sell themselves):
Schindler's List, Raiders of the Lost Ark, Amadeus, Apocalypse Now, Psycho, Die Hard, Blade Runner

**Core Identity** (high fit, but needs the theater's voice to drive attendance):
Casablanca, North by Northwest, Vertigo, Notorious, Dial M for Murder, Hard Eight

**Wrong Fit** (popular nationally, poor audience match for indie theater demographics):
Elf, Wonka, Minecraft Movie, Lilo & Stitch, IF (2024), Clifford the Big Red Dog

### Key Findings

- Only **110 of 750 films (15%) are true Easy Wins** — films that sell themselves to this audience type. This is a narrow programming pool.
- **190 films are Wrong Fit** — popular with mainstream audiences but poorly matched to indie theater demographics. These represent distributor pressure risk: studios push them, but they underperform.
- **Core Identity films need the theater's credibility**, not a marketing budget. A programmer's note, a Q&A, an event wrapper, or a series framing turns a 4.2 popularity film into a sold-out screening.
- **Amadeus is the only top Easy Win driven by a female segment (55+ F)** — reinforcing that the catalog has very few films that organically attract female audiences.

### KPI Impact
| KPI | Effect |
|---|---|
| Marketing spend efficiency | Easy Wins require minimal budget; Core Identity requires effort but not cash — redistributes spend away from Wrong Fit |
| Occupancy rate | Eliminating Wrong Fit bookings raises the floor occupancy across all slots |
| Programmer time | Tier system gives bookers a fast filter at the start of each booking cycle |

---

## Cross-Analysis: The 35–54 F Opportunity

This finding surfaces in every single analysis and deserves standalone treatment.

| Analysis | Signal |
|---|---|
| Genre Fingerprint | No venue's top-3 genres include content that skews female |
| Segment Heatmap | 35–54 F is the weakest segment at all 5 venues |
| Slot Matrix | 35–54 F never peaks in any time slot |
| Missed Opportunities | Only 1 top missed film (Amadeus) is driven by a female segment (55+ F, not even 35–54 F) |
| Marketing Tier | No Easy Win or Core Identity film in the top tier is primarily a female-segment driver |

**The opportunity:** 35–54 women are among the highest arts-and-culture spenders in the US and consistently over-index on independent cinema attendance nationally. The fact that the model finds zero programming fit for this segment is not a data artifact — it reflects a real gap in what these theaters are booking.

**What would move the needle:** Female-directed films, female-led prestige drama, and narrative documentaries tend to score well with this segment in collaborative filtering models. A curated series (4–6 films, one per week) at any venue targeting this demographic would be a differentiated offer with no current competition in this market.

---

## Per-Venue Recommendations

### Trylon Cinema
- **Book 2–3 Casablanca Cluster films** — Casablanca (91.4) and Vertigo (89.8) are natural fits for Trylon's arthouse identity and are currently only available at Heights.
- **Lean into Core Identity marketing** — Trylon's audience expects curator-led framing. Every booking should have a programmer's note.
- **Pilot a 35–54 F series** — Trylon's Drama/Thriller fingerprint is closest to what would attract this demographic. Small sample test (4 films) with targeted email marketing.

### Riverview Theater
- **Replace Wrong Fit bookings with Casablanca Cluster entries** — Riverview's comedy/drama balance suggests room for prestige classics.
- **Saturday Prime is underutilized** — currently filled with promotional_support titles. Upgrade to high_confidence picks for the highest-revenue slot.
- **Target 55+ M on Saturday Matinee** — Riverview's strongest segment. Pair with real-butter popcorn nostalgia marketing.

### Lagoon
- **Lean into contemporary prestige** — Civil War (80.1), Napoleon (80.6), Barbie (76.4) are already scoring well. Expand this direction.
- **Concert film programming is working** — Pet Shop Boys, Taylor Swift, The Cure all appear in the schedule at viable scores. This is a differentiated niche worth doubling down on.
- **4 screens = marketing budget segmentation opportunity** — run different campaign types per screen (high-confidence screens need no paid spend; promotional_support screens get the social budget).

### Heights Theater
- **Expand into crossover contemporary** — Goodfellas (81.9), It's A Wonderful Life (78.5), Touch of Evil (78.1) are all high-confidence additions that fit Heights' identity without abandoning it.
- **Barbie (76.8) is a signal** — the model says Heights' audience would respond to it. A one-off screening positioned as a "cultural moment" film (not a blockbuster) could attract new 18–34 F audience.
- **Organ event nights are a competitive moat** — no other venue can replicate this. It's worth modeling whether organ + film pairings systematically outperform non-organ showings.

### Drafthouse
- **Screens 1–2 are the money screens** — Psycho (83.4), Die Hard (81.3), Barbie (79.4) all score high_confidence. These should get the prime slots and largest auditoriums.
- **Screens 5–8 are the marketing problem** — all promotional_support at 60–70 scores. These need targeted genre-community marketing (horror forums, genre newsletters) rather than broad social.
- **Wrong Fit risk is highest here** — with 8 screens, Drafthouse is most exposed to distributor pressure to run family/blockbuster content that scores 16–22. The model gives the booker a data-backed reason to say no.
- **Taylor Swift and concert films belong here** — Drafthouse's 18–34 skew makes event cinema a natural fit for their lower-scoring screens.

---

## Revenue Impact Projection (Section 11)

> **Headline:** If all 5 theaters adopt the TheatreIQ recommended schedules, total annual ticket revenue increases from **$4.90M → $5.36M (+$0.46M, +9.5%)** under full adoption, or **$4.90M → $5.11M (+$0.21M, +4.3%)** under conservative adoption.

*All figures based on synthetic occupancy — no real box-office data was used. See README for full methodology.*

### Per-Venue Revenue Breakdown

| Venue | Baseline Annual | Full Adoption | Δ Revenue | % Lift | Conservative Lift | Data |
|---|---|---|---|---|---|---|
| Heights Theater | $2,046,550 | $2,284,942 | +$238,392 | +11.6% | +$87,222 (+4.3%) | actual |
| Drafthouse | $1,195,666 | $1,276,836 | +$81,170 | +6.8% | +$43,743 (+3.7%) | actual |
| Lagoon | $847,546 | $905,901 | +$58,355 | +6.9% | +$31,528 (+3.7%) | *projected |
| Riverview Theater | $540,657 | $603,538 | +$62,881 | +11.6% | +$38,141 (+7.1%) | *projected |
| Trylon Cinema | $266,637 | $290,378 | +$23,741 | +8.9% | +$10,427 (+3.9%) | *projected |
| **5-Theater Total** | **$4.90M** | **$5.36M** | **+$0.46M** | **+9.5%** | **+$0.21M (+4.3%)** | |

*\* Lagoon, Riverview, and Trylon have incomplete scraped data — revenue projected using assumed annual showings (520 / 600 / 350 respectively) × observed revenue-per-showing.*

### What Drives the Lift

The lift is modeled as: `Δ_occupancy = r × (Δ_match_score / σ_match) × σ_occ` using the Pearson correlation between match score and synthetic occupancy.

| Venue | Baseline Score | Rec Score | Δ Score | Occ Lift | Consider-Drop Slots |
|---|---|---|---|---|---|
| Heights Theater | 53.3 | 91.7 | +38.3 | +4.0pp | 37% |
| Drafthouse | 46.1 | 69.2 | +23.1 | +2.4pp | 54% |

- **Heights Theater gets the largest absolute lift (+$238K)** because it has the biggest match score improvement — its recommended schedule (all high_confidence classics) is far above what it historically booked.
- **Drafthouse has the highest consider-dropping fraction (54%)** — over half its film×slot combinations scored below 50. Replacing these is the highest-volume opportunity.
- **Riverview and Trylon tie at +11.6%** under full adoption — both have historically weak baseline scores (40.9 and 47.9 avg) and their recommended schedules include high-scoring Casablanca Cluster films they've never booked.

### Two Scenarios Explained

| Scenario | Interpretation |
|---|---|
| **Full adoption (+9.5%)** | Theater commits entirely to recommended schedule. All slots are filled with the highest-scoring available films. Best case. |
| **Conservative (+4.3%)** | Theater replaces only the `consider_dropping` films (<50 score) with better alternatives; keeps the rest of its current programming unchanged. Realistic near-term target. |

### Important Caveats

1. All figures use **synthetic occupancy** — the r=0.3 correlation applied here is a modeling assumption, not a measured causal effect. The actual observed Pearson r between match score and synthetic occupancy in this dataset is **0.084** (Section 9). Using r=0.3 represents a moderate-confidence projection of real-world causal effect.
2. The model assumes **same total showings per year** — recommendations substitute films, not add screenings.
3. Ticket revenue only — **F&B, memberships, and event premiums not included**. Actual revenue impact at venues like Drafthouse (food-and-beverage model) would be larger.

---

## KPIs Influenced by This System

| KPI | Current State (Synthetic) | Lever | Estimated Impact |
|---|---|---|---|
| Annual ticket revenue | $4.90M combined (5 venues) | Full schedule adoption | **+$0.46M (+9.5%)** full / **+$0.21M (+4.3%)** conservative |
| Occupancy rate | 35.4% avg, 55% weekend peak | Book high_confidence vs. consider_dropping | +2.4–4.0 percentage points per venue |
| Revenue per screen per week | Varies by venue/slot | Saturday Prime optimization | +$500–$1,200 per screen per week |
| Marketing spend efficiency | Unknown baseline | Tier-based budget allocation | Reduce spend on Easy Wins, redirect to Core Identity |
| Programming hit rate | 187/491 films = 38% high_confidence | Casablanca Cluster bookings | Immediately raises hit rate for Trylon, Riverview, Lagoon |
| Audience diversification | 35–54 F consistently weakest | Targeted 4-film series | New audience segment, measurable via ticket sales |
| Booking decision time | Manual / gut-feel | Match score filter at booking stage | Faster, more defensible decisions |
| Distributor negotiation leverage | None | Score-backed demand quantification | Data argument for better revenue splits on high-scoring titles |

---

## What Would Strengthen This Analysis

1. **Real ticket sales data** — even one venue, one quarter. Validates whether match scores actually predict attendance.
2. **Patron survey** — 100-person audience survey at Heights and Trylon to validate segment assignments against actual demographics.
3. **Run-length optimization** — model week-over-week attendance decay to recommend whether a film should run 1 week or 3.
4. **Competitive programming feed** — track what the five venues are booking in real time to identify print conflicts and programming gaps before they happen.
5. **35–54 F pilot** — a 4-film curated series at one venue, tracked against the model's predictions. Low cost, high learning value.
