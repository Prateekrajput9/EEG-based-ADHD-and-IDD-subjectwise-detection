# Panel Q&A: likely questions and answers

<!-- docs/panel_qa.md is generated from docs/panel_qa.template.md by
     `python report/build_final.py`. Edit the template, not the output.
     Double-brace placeholders are filled from results/; ?? means not computed yet. -->

> Research prototype. Not a diagnostic tool. DS-2 caveat: the ADHD children were on
> methylphenidate (Ritalin) and the controls were not.

**1. The paper reports 100% and 99.17%. Did you reproduce it?**
Yes, under the paper's protocol. My code gives 99.69% on DS-1 rest and
85.81% on DS-2 with 10-fold cross-validation over 5-second windows. Repeated runs
give identical metrics.

**2. Then why do you say it doesn't generalise?**
In window-wise cross-validation, windows of the same child are in both the training and the
test set, so the model can recognise the child instead of the disorder. When I hold out
whole children, the same model gives 62.73% (95% CI
44.1–79.7%) on DS-1 rest and 59.30% (CI 56.3–62.2%) on
DS-2. That is the number that matters for a new child.

**3. How do you know there is no leakage in your own evaluation?**
Every split is by child: leave-one-child-out on DS-1, stratified grouped 10-fold on DS-2.
Scaling, mRMR, channel selection, tuning and early stopping are fitted inside the training
fold. An automated test checks that no child appears in both parts of any fold, including the
inner tuning folds. The test suite runs with `python -m pytest tests`.

**4. Why bootstrap over children and not over windows?**
Windows of the same child are not independent. The unit we want to generalise to is the
child, so the confidence interval resamples children.

**5. 14 participants is very small. What can you conclude from DS-1?**
Little with confidence. One participant is 7 percentage points of participant-level accuracy,
and the participant-level CI for the reproduced model on DS-1 rest is
50.0–92.9%. Even 14 out of 14 correct only gives an exact 95% interval of
76.8–100%. I report intervals everywhere so that differences inside them
aren't over-read.

**5b. Are the DS-1 participants children?**
No. The dataset's own metadata gives the IDD participants' ages as
26–31 years, and it gives no ages for the controls. The base
paper presents its study as detection in children, but that holds only for DS-2 (ages 7–12).
In my report "child-level" is a metric name meaning "per subject".

**6. Which method worked best under the strict protocol?**
On DS-1 rest: MiniRocket + ridge classifier, at 99.8% balanced window accuracy
(CI 99.5–100.0%), with 14/14 children correct. On DS-2:
Riemannian tangent space (OAS covariance) + SVM, at 72.6% (CI 67.4–77.2%), with
92/121 children correct. I report all 24 methods, not only the best.

**7. If a simple model does that well on DS-1, isn't the problem solved?**
Not necessarily. The groups differ strongly in their recordings: one number per rest
recording, relative beta power, separates all 14 (IDD 0.139–0.286, controls
0.040–0.134). Raised beta could be the disorder, but also muscle tension, recording
conditions (session, electrode contact, room) or age, and these data cannot tell them apart.
One control recording also contains a large artefact (10 times the median
amplitude). It needs an independent cohort recorded with the same protocol for both groups.

It is not a leak in my protocol. ShallowConvNet gets 99.7% on DS-1 music with
leave-one-subject-out. With the participants' labels randomly permuted and everything else
identical, it gets 42.1–47.7%, which is chance. That used two permutations, so
it is a sanity check, not a significance test.

**8. Why didn't hyperparameter tuning help?**
Tuning was nested, inside each training fold. With 13 training children the chosen settings
change from fold to fold. Nested-tuned KNN gives 43.8% on DS-1 rest,
versus 62.73% with the paper's Table II settings. The Table II settings
were chosen on all the data, so they are not a fair held-out comparison either.

**9. Is the DS-2 result about ADHD?**
It can't be separated from medication: the ADHD children took methylphenidate and the
controls didn't. The result is an "ADHD on medication versus unmedicated control" comparison.
The groups also behaved differently: the task ran until the child finished, and ADHD children
took longer (60.3 versus 48.5 windows on average).

**9b. You picked the best of many methods. Isn't that optimistic?**
Yes, slightly: the best method was chosen on the same test folds it is scored on. That is why I
report all methods and treat the paper's fixed Table II model and the nested-tuned KNN as the
unbiased comparisons. A fully clean number would need the method choice inside the training
folds too.

**10. How much does the split choice matter on DS-2?**
Over 20 random assignments of children to folds, subject-wise accuracy is
59.0 ± 0.5% (range 57.7–60.4%). The
mid-semester number (57.54%) came from an unstratified, unseeded split.

**11. Does SMVMD itself add anything?**
The ablation ladder (raw → fixed bands → VMD → MVMD → SMVMD, same nine features each) answers
this; see the report's ablation table. In the reproduction code, raw-signal features scored at
least as well as SMVMD on DS-1 rest subject-wise.

**12. Can a model trained on one dataset work on the other?**
I tested it on the 10 electrodes both montages share. The datasets differ in disorder,
headset and task, so this is a domain-shift test. A classifier that only has to tell which
*dataset* a window comes from gets 99.8% balanced accuracy, so the
datasets differ strongly before any disorder signal is considered.

**13. What does the model look at?**
Out-of-fold SHAP for the spectral XGBoost model ranks logpow_beta, relpow_beta, hjorth_activity highest on DS-1
rest. Importance shows what the model uses in these data. It is not a mechanism, and on DS-2
it may reflect medication.

**13b. What about the theta/beta ratio?**
Scored with the direction fixed in advance (higher = disorder), it gives a window AUC of
0.42 (CI 0.36–0.49) on DS-2. That's below 0.5: the ADHD children have a
*lower* ratio than the controls, the opposite of the classic hypothesis. Stimulant medication is
reported to lower it, so this fits the medication confound, but the data can't prove that. On
DS-1 rest it gives 0.38 (CI 0.18–0.60), which is chance.

**14. Is the live demo diagnosing me?**
No. The demo model was trained on 14 participants recorded with one headset, from cleaned
recordings. A new person on another headset and in another session is outside its training
data, so the output has no diagnostic meaning. The demo shows
that the pipeline runs in real time and gives exactly the same windows and features as the
offline code. An automated equivalence test checks this.

**15. What would you do next?**
Get an independent cohort recorded with the same protocol for both groups, with medication
status controlled. Report subject-wise results with child-level CIs as standard and publish
the split definitions. Add artefact rejection and calibration to the live system before any
trial with children, under ethical approval.
