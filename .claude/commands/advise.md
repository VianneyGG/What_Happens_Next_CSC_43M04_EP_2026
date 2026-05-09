## /advise

Search the experiment registry for prior runs relevant to the current goal.

Usage: `/advise <keyword>` (e.g. `/advise lstm`, `/advise num_frames`, `/advise overfitting`)

### Steps

1. `ls .claude/experiments/ 2>/dev/null | grep -v '^_' | head -20`
2. `grep -r "$ARGUMENTS" .claude/experiments/ --include="*.md" -l 2>/dev/null | head -10`
3. For each matching file: read its **Results**, **What Worked**, and **Failed Attempts** sections.
4. Summarize in 3-5 bullets:
   - What configs/approaches were verified
   - What failed and why (from the Failed Attempts table)
   - Exact hyperparameters that worked
5. If no matches: "No prior experiments found for '$ARGUMENTS'. Run /retrospective after your session to seed the registry."
