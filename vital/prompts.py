SUBCLAIM_QUERY_IMPORTANCE = '''You are performing step two of a four part fact-checking process:
(1) Decompose a paragraph into individual claims.
(2) Given a query and set of claims, rank by decreasing query-importance (this step).
(3) Check the correctness of each claim.
(4) Score the paragraph, weighting by importance.
This step is completely independent of factual correctness, and only focuses on the query-importance of claims for answering the query. Even factually incorrect claims should be ranked highly if they directly answer the query.

Instructions: You are provided with a query and set of claims. Rank the claims in decreasing order of query-importance. A claim exhibits high query-importance when it addresses a central aspect of the query, and low query-importance when it contributes only peripheral or background information. Rank claims independent of correctness, instead only based on query-importance. A later step will check for correctness of claims.

Assign query-importance labels using exactly these three categories:
- "vital" - Essential claims that directly address the core query
- "okay" - Supporting claims that provide useful but non-essential information
- "less-important" - Background or tangentially related claims with minimal relevance

Ordering Rules:
- All "vital" claims must appear first, then all "okay" claims come second, and "less-important" claims come last.
- Within each category, order by decreasing importance.
- If two or more claims address the same aspect of the query, keep them grouped in the order they appear, even if their answers contradict. For example:
    ...
    [S3] Washington, D.C. is the capital of Canada.: "vital"
    [S8] Washington, D.C. is the capital of the United States.: "vital"
    ...
- Do not adjust rankings based on factual correctness, this will be handled in step 3.

Output Format:
[Claim ID] <claim text>: "label"
[Claim ID] <claim text>: "label"
...

Requirements:
- Label every claim exactly once
- Use only the three specified labels
- Maintain the original claim count
- Return only the labeled, ordered list (no explanations)
Below is your task.

###Your task:
Query: [QUERY]
Claims:
[SUBCLAIMS]
Ranked Claims:'''


# Nugget alignment step
# Set N are nuggets and Set S are subclaims
NUGGET_SUBCLAIM_ALIGNMENT = '''You are given two sets of subclaims, Set N and Set S, each containing individual statements derived from a paragraph. Your task is to align subclaims in Set N to semantically equivalent or closely related subclaims in Set S.
A subclaim in set N may:
- Match exactly one subclaim in Set S (1:1 alignment)
- Match multiple subclaims in Set S (1:many)
- Not match any subclaim in Set S (no alignment)
Subclaims in Set S can only be aligned with one subclaim in Set N. Some subclaims in Set S may not align to any subclaims in Set N.
For each subclaim in Set N:
- Identify all subclaims in Set S that express the same idea, a paraphrase, or a logically entailed version of the subclaim OR express the opposite idea, directly contradicting the subclaim in N.
- Record the alignment(s) as pairs: (Ni, Sj), (Ni, [Sj, Sk]) or note (Ni, None) if no match exists.
- Ignore minor surface differences, focus on core semantic equivalence, entailment, or contradiction.
Guidelines:
- Use alignment only if the meanings are substantively overlapping.
- Prefer conservative alignment: don't align vague or only tangentially related subclaims.
You are first given an example, and then your task.

###Example:
#Set N
N1: Swallowing watermelon seeds is harmless.
N2: Watermelon seeds pass through the digestive system undigested.
N3: Accidentally swallowing seeds is not a concern.
N4: Watermelon seeds are excreted in the stool.
N5: Watermelon seeds are nutritious if chewed or sprouted.
N6: Roasted watermelon seeds provide protein and healthy fats.
N7: Watermelon seeds contain minerals like magnesium.
N8: Watermelon seeds are black in color.
#Set S
S1: The watermelon seeds pass through your digestive system.
S2: The watermelon seeds do not cause harm to your digestive system.
S3: There are popular myths about swallowing watermelon seeds.
S4: One popular myth is that swallowing watermelon seeds leads to a watermelon growing in your stomach.
S5: Swallowing watermelon seeds does not lead to a watermelon growing in your stomach.
S6: The seeds are generally harmless.
S7: The seeds will move through your gastrointestinal tract.
S8: The seeds will be excreted in your stool.
S9: Watermelon seeds are nutritious.
S10: Watermelon seeds contain protein.
S11: Watermelon seeds contain healthy fats.
S12: Watermelon seeds contain magnesium.
S13: Watermelon seeds contain other beneficial nutrients.
S14: To access these nutrients, the seeds need to be chewed.
S15: To access these nutrients, the seeds need to be sprouted.
S16: Swallowing the seeds whole means they will likely pass through your system undigested.
S17: In many cultures, watermelon seeds are roasted.
S18: In many cultures, watermelon seeds are consumed as a snack.
S19: Watermelon seeds are consumed as a snack similar to pumpkin seeds.
S20: Watermelon seeds are consumed as a snack similar to sunflower seeds.
S21: Eating them in this form can be a healthy addition.
S22: Eating them in this form can be a healthy addition to your diet.
S23: If you accidentally swallow a few watermelon seeds, there is no need for concern.
S24: Swallowed watermelon seeds will pass through your digestive system.
S25: Swallowed watermelon seeds will pass without any adverse effects.
S26: Swallowing a watermelon seed is very harmful.
#Alignments
(N1, [S6, S25, S26])
(N2, [S1, S2, S7, S16, S24])
(N3, S23)
(N4, S8)
(N5, [S9, S14, S15])
(N6, [S10, S11])
(N7, S12)
(N8, None)

###Your task:
#Set N
[NUGGETS]
#Set S
[SUBCLAIMS]
#Alignments'''


# Batch evaluation against grounding
EVAL_PROMPT = '''You are trying to verify how factual a list of claims are, given provided evidence. All necessary specific details—including entities, dates, and locations—must be explicitly named, and verify here means that every detail of a claim is directly confirmed by the provided evidence. The verification process involves cross-checking each detail against the evidence; a detail is considered verified if it is clearly confirmed by the evidence.

For each claim, classify it as follows:
- Supported: Every detail of the claim (including entities, dates, and locations) is directly confirmed by the provided evidence with no contradictions.
- Unsupported: One or more details of the claim are either missing from or contradicted by the provided evidence, even though the claim remains verifiable using external sources.

Output format:
[S1] <claim 1>: <your judgment of fact 1>
[S2] <claim 2>: <your judgment of fact 2>
...
[Sn] <claim n>: <your judgment of fact n>

You do not need to justify your judgment. Respond with the list of claims and judgments and nothing else.

###Evidence:
[EVIDENCE]

###Claims:
[SUBCLAIMS]

###Output:'''
