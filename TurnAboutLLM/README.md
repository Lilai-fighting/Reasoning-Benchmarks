# The Turnabout LLM Dataset

## Coverage

The **Turnabout LLM** dataset contains 9 installations of classic detective visual novel games:
- (AA1) Phoenix Wright: Ace Attorney 
- (AA2) Phoenix Wright: Ace Attorney – Justice for All 
- (AA3) Phoenix Wright: Ace Attorney – Trials and Tribulations
- (AA4) Apollo Justice: Ace Attorney
- (AA5) Phoenix Wright: Ace Attorney – Dual Destinies
- (AA6) Phoenix Wright: Ace Attorney – Spirit of Justice
- (GAA1) The Great Ace Attorney: Adventures
- (GAA2) The Great Ace Attorney 2: Resolve
- (DGRP1) Danganronpa: Trigger Happy Havoc
> Note: Danganronpa 2: Goodbye Despair and Danganronpa V3: Killing Harmony were not included due to the lack of a well-structured transcript. Let us know if you know those that exist!

The dataset comes in a collection of JSON files. The ready-to-use datasets are found at [[Ace Attorney dataset]](aceattorney_data/final/) and [[Danganronpa dataset]](danganronpa_data/final/).

## Organization

A case (e.g., 1-2: Turnabout Sisters in AA1) is typically split into multiple episodes (e.g., 1-2-2: the second day, or the court day, of 1-2). Each episodes is represented with a JSON file (e.g., [1-2-2.json](aceattorney_data/final/1-2-2_Turnabout_Sisters.json)). Each JSON file has the following schema:
- `previousContext`: The story that has happened until the start of this episode. The story is expressed as a long string containing all transcripts, flattened.
- `characters`: Information about all involved characters in the case
    - `currentChapter`, `name`, `age`, `gender`
    - `description1`, `description2`, ...: case-specific descriptions extracted from the wiki source. Multiple descriptions happen when the character description is updated through the progression of the game. For simplicity, we concatenate all descriptions in an case and use that throughout the evaluation.
- `evidences`: Information about all involved evidences in the case
    - `currentChapter`, `name`, `type`, `obtained`
    - `description1`, `description2`, ...: same as above; case-specific descriptions extracted from the wiki source. Multiple descriptions happen when the evidence description is updated through the progression of the game. For simplicity, we concatenate all descriptions in an case and use that throughout the evaluation. Somtimes, the evidence may contain additional information that requires clicks to reveal; this is copied over with the tag `[info]`. Or, the evidence may contain one or more critical images; they are captioned with the tag  `[caption]`
- `turns`: The cross-examination (AA) and class trial (DGRP) portions of the games, where the player or model needs to take actions.
    - `category`: this can only be "cross_examination", since the rest of the actionable gameplays are excluded in this dataset
    - `newContext`: the story in a long string from the start of the previous turn to the start of the current turn
    - `testimonies`: the testimonies given by one or more witnesses; one or more of them contains a contradiction with one or more evidences
        - `testimony`: the text of the testimony; in rare cases where the testimony comes with an image, it is captioned with the tag `[caption]`
        - `person`: name of the witness giving the testimony
        - `present`: the list of correct evidences to be presented that contradict this testimony, or empty if the testimony has no issue. In AA2 and AA3 only, characters can also be presented. For simplicity, we only consider evidences to present
        - `source`: a collection of cited sources to explain the contradiction
            - `evidence_span`: the span of the evidence (all strings under `evidences` concatenated) that critically constitutes the contradiction
            - `testimony_span`: the span of the testimony that critically constitutes the contradiction
            - `explanation`: an annotated free-form explanation of the contradiction
            - `is_self_contained`: whether the contradiction be be deduced with only this pair of a testimony and an evidence
            - `context_span`: empty if `is_self_contained`; else, the span from anywhere else in this file such as `newContext`, `previousContext`, other `testimony`, other information from `evidences`, etc.
    - `noPresent`: Whether this turn does not have a contradiction and should be skipped in evaluation. It is set to true automatically if the turn is so in the game (usually requires pressing some testimonies, an action not considered in this dataset, and moves on to the next turn), or manually if the contradiction is not rigorous enough
    - `summarized_context`: A minimal summary of the story so far, filling in the gaps of any missing context annotated in `context_span`. The information here should make the contradiction fully self-contained.
    - `labels`: Manually annotated labels of the most critical reasoning type involved in finding the contradiction
        - **numerical**: a present is labeled as "numerical" if and only if the core contradiction is a difference in some numbers (e.g., "I heard 1 gunshot" vs. "2 gunshots were fired"); it should not be labeled so if numbers are mentioned but do not consitute a contradiction (e.g., "I gave the victim 2 items" vs. "The witness never met the victim")
        - **temporal**: a present is labeled as "temporal" if and only if the core contradiction involves time (e.g., "He died before noon" vs. "Time of death is after 3PM"); this commonly stack with other labels such as "numerical"; it should not be labeled so if time is mentioned but do not consitute a contradiction (e.g., "I met the victim in the morning" vs. "The witness never met the victim")
        - **spatial**: a present is labeled as "spatial" if and only if the core contradiction involves space (e.g., "I killed him at the bus stop" vs. "The victim was found dead in his home"); it should not be labeled so if space is mentioned but do not consitute a contradiction (e.g., "I met the victim at school" vs. "The witness never met the victim")
        - **object property**: a present is labeled as "object property" if and only if the core contradiction involves certain non-universal properties of an object (e.g., "I saw him beaten by a club" vs. "Autopsy report shows only trauma of piercing", because a club does not cause piercing damage); this object cannot be time or space, but can be abstract entity such as a concept (e.g., "I never told anyone this idea" vs. "The victim wrote down this idea in a notebook"); it should not be labeled so if a human's *behavior* is involved, and should be instead labeled as "behavior" (see below); at times, this may stack with other labels (e.g., "I did not hear anything from the clock at noon" vs. "The clock sounds at noon" can also be labeled as "temporal", because the mention of the time "noon" is equally core to the contradiction) (e.g., "I saw the vase" vs. "There was a wall between the witness and the vase" can also be labeled as "spatial", because the positioning of the wall and the visibility of the vase are equally core to the contradiction)
        - **behavior**: a present is labeled as "behavior" if and only if the core contradiction involves a human's behavior, either pertaining to their intent, habits, or preference (e.g., "Larry hates music" vs. "Larry is reported to listen to music every day"); extenuating exceptions are not considered unless there is strong evidence otherwise; it should not be labeled so if the contradiction is labeled as something else above, leading to a derivative contradiction in behavior (e.g., "I killed him at the bus stop" vs. "The victim was found dead in his home" may lead to a corollary of "I killed him" vs. "I cannot have killed him")
        - **spelling**: e.g., "Harry" vs. "Henry"
    - `reasoning`: Manually annotated list of **facts** or **propositions** to arrive at the contradiction. Considerations:
        - A fact is a paraphrase of the `testimony_span` (e.g., I saw the victim getting shot), `evidence_span` (e.g., Only piercing wounds were found.), or `context_span`
        - A proposition is some general rule of implication (e.g., If someone gets shot, there will be ballistic wounds, not piercing wounds.). No matter how obvious, there is always at least one proposition. A proposition is framed as Modus ponens (assertion P + conditional if P then Q → assertion Q)
        - Each fact should be an atomic unit, though subjectivity is inevitable