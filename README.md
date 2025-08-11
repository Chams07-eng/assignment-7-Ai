# assignment-7-Ai
Ethics
Q1: Define algorithmic bias and provide two examples of how it manifests in AI systems.
Definition: Algorithmic bias occurs when an AI system produces systematically unfair outcomes that disadvantage certain individuals or groups. This can arise from biased training data, model design choices, skewed sampling, or deployment context.
Examples:
Hiring systems trained on historical hiring data that favored men can learn to penalize resumes with indicators of female gender (e.g., women’s colleges, maternity gaps).

Facial recognition models trained on datasets dominated by lighter-skinned faces tend to have much higher error rates on darker-skinned individuals, causing misidentification.

Q2: Explain the difference between transparency and explainability in AI. Why are both important?

Transparency refers to the openness about an AI system’s components: data sources, model type, training process, performance metrics, and governance. A transparent system reveals what went into building it.

Explainability (or interpretability) refers to how well a model’s decisions can be understood and justified — e.g., providing reasons or feature importance for a specific prediction.
Why both matter: Transparency lets stakeholders audit and hold developers accountable; explainability helps affected individuals, regulators, and operators understand and contest decisions. Together they enable trust, debugging, and compliance with legal/ethical standards.

Q3: How does GDPR impact AI development in the EU?
Key GDPR implications for AI:

Lawful basis & purpose limitation: You must have a lawful basis (consent, contract, legitimate interest, etc.) for processing personal data, and use data only for specified purposes.

Data minimization & storage limitation: Collect and keep only data necessary for the purpose.

Rights of data subjects: Right of access, rectification, erasure (“right to be forgotten”), and data portability — all can affect models and pipelines.

Automated decision-making: Article 22 restricts solely automated decisions that produce legal or similarly significant effects; individuals may have a right to meaningful information about the logic and to contest decisions.

Accountability & DPIA: Data Protection Impact Assessments (DPIAs) are required for high-risk processing (often applies to large-scale profiling or biometric systems). Controllers must demonstrate compliance and maintain records.

Privacy by design/default: Integrate privacy protections into systems from the start (e.g., pseudonymization, access controls).

(These are practical effects: modelers must plan for consent, logging, DPIAs, mechanisms to explain/contest decisions, and careful data governance.)

2. Ethical Principles Matching
Match definitions to letters:

Ensuring AI does not harm individuals or society. — B) Non-maleficence

Respecting users’ right to control their data and decisions. — C) Autonomy

Designing AI to be environmentally friendly. — D) Sustainability

Fair distribution of AI benefits and risks. — A) Justice

Part 2 — Case Study Analysis (40%)
Case 1: Biased Hiring Tool (Amazon scenario)
Scenario recap: Amazon’s AI recruiting tool penalized female candidates.

(a) Identify sources of bias
Training data bias: Historical hiring decisions reflected male-dominated hires, so the model learned to rank male-like features higher.

Label bias / proxy features: Labels (e.g., past-hire = good candidate) reflect prior human bias. The model may use proxies (e.g., words like “women’s college” or career gaps) that correlate with gender.

Sampling bias: Under-representation of female applicants in training data; imbalance leads to poor generalization.

Objective / loss function mis-specification: Optimizing only for historical hiring outcomes without fairness constraints enshrines bias.

Feature engineering choices: Including features that leak gender or socio-cultural proxies without mitigation.

(b) Three fixes to make the tool fairer
Data-level corrections:

Gather a balanced dataset (oversample female applicant cases where feasible) or use reweighting techniques to correct historical imbalance.

Remove or mask direct gender indicators and known proxies; but be careful—blindness alone isn’t sufficient if proxies persist.

Algorithmic fairness interventions:

Apply fairness-aware learning (e.g., adversarial debiasing, equalized odds constraints, or optimized post-processing) to enforce parity across protected groups.

Use techniques like reweighing, disparate impact remover, or constrained optimization that trade off minimal accuracy loss for fairness gains.

Human-in-the-loop & process changes:

Make the tool assistive (ranking suggestions) rather than automatic. Require diverse human panels for final decisions and regularly audit model outputs.

Add an explainability module so hiring staff can see why candidates were ranked as they are.

Enact governance: logging, feedback loops to capture mistakes, and a process to update model with corrected labels.

(c) Metrics to evaluate fairness post-correction
Demographic parity / Statistical parity difference (difference in positive selection rates between groups).

Equalized odds (compare true positive rates and false positive rates across groups).

Predictive parity / calibration (are predicted scores calibrated across groups?).

False negative rate parity (important for hiring: missing qualified members).

Disparate impact ratio (selection rate for protected group divided by unprotected group; e.g., 80% rule).

Subgroup performance metrics: precision, recall, AUC per group.

Business + fairness dashboard: track hiring conversion rates by group, time-to-hire, and downstream performance (if ethically and legally permissible).

Case 2: Facial Recognition in Policing
Scenario recap: Misidentification rates are higher for minorities.

(a) Ethical risks
Wrongful arrests / injustice: Higher false positives for minorities can lead to wrongful stops, arrests, and serious legal harms.

Discrimination & civil liberties: Systemic targeting increases profiling and exacerbates distrust between communities and law enforcement.

Privacy violations: Mass surveillance and face-matching without consent invasive of personal privacy.

Chilling effects: Public may avoid lawful activities due to fear of surveillance.

Due process & transparency: Lack of explainability and opaque use undermines rights to contest evidence.

Mission creep: Systems deployed for narrow use expand to more intrusive applications without oversight.

(b) Recommended policies for responsible deployment
Use restrictions & risk classification: Ban or restrict face recognition for high-risk uses (e.g., identifying suspects in crowds) unless strict standards met. Allow limited, well-justified uses subject to oversight.

Regulatory oversight & DPIAs: Require independent Data Protection Impact Assessments and external audits before deployment.

Human oversight & corroboration: Never allow automated identification to be sole basis for enforcement action — require corroborating evidence and human verification.

Transparency & documentation: Publicly disclose use cases, data retention policies, performance metrics disaggregated by demographic groups, and mechanisms to challenge matches.

Performance & fairness thresholds: Enforce minimum accuracy and disparity thresholds across demographic groups; if not met, prohibit deployment.

Consent & notice (where feasible): Give notice where surveillance is in public/private spaces and clearly define retention and deletion policies.

Logging & accountability: Maintain immutable logs for matches and actions taken, with accessible redress mechanisms and independent audits.

Community & legal oversight: Engage affected communities and legislate clear boundaries (e.g., parliamentary or municipal approval, judicial warrants for sensitive uses).
