## 🧠 What is *raink*?

**raink** is an open-source, command-line tool developed by Bishop Fox that uses Large Language Models (LLMs) to perform document or item ranking through a novel **listwise ranking algorithm**. Unlike traditional methods that rely on pairwise comparisons (which can be computationally expensive), *raink* enables **scalable, consistent ranking** using **multi-sample sort**, reducing the unpredictability inherent in LLMs.

---

## 🎯 Why *raink* was built

LLMs are great at reasoning but **struggle with reliable ranking**:

* Often produce **incomplete, inconsistent, or hallucinated lists**.
* Pairwise comparisons (like tournament brackets) are accurate but **computationally costly** — O(N²) or O(N log N) LLM calls.

*raink* solves this with a listwise approach that:

* Uses **batch comparisons** of shuffled item groups,
* Performs **multiple random passes** to average out inconsistencies,
* **Recursively refines** the top results — all with **linear scalability**.

---

## ⚙️ How *raink* works (Multi-Sample Sort)

1. Items (e.g., documents, code functions) are **shuffled and grouped** into batches.
2. The LLM ranks each batch; each item receives a **relative score**.
3. This process is repeated across **multiple shuffled passes**.
4. Scores are **averaged**, and the top-ranked items undergo a **refined re-ranking**.
5. The final result is a **stable, statistically sound ranked list**.

---

## 🔐 Cybersecurity Use Case: End-of-Day Vulnerability Analysis

In cybersecurity, *raink’s* **multi-sample sort** algorithm significantly improves how LLMs rank potentially vulnerable code segments after a vulnerability has been disclosed.

### 📌 The Problem

When a new vulnerability is reported (e.g., via CVE or GitHub advisory), security teams often review the corresponding codebase to determine **what changed** and **where the fix was applied**. With hundreds of functions to inspect, using an LLM to locate the fixed function can help — but LLMs are **non-deterministic** and may produce **different rankings each time**.

### ✅ The Solution

*raink* solves this with **multi-sample sort**:

* It repeatedly shuffles and re-evaluates subsets of functions.
* By **averaging across runs**, it compensates for LLM randomness.
* This **stabilizes the output**, directing analysts to the most likely fixed functions.

### 🎯 Real Result

In a test involving **293 functions**, *raink* correctly ranked the fixed function within the **top 9%**, significantly reducing the time and effort needed by human analysts.

This approach makes end-day triage **faster, more consistent, and more actionable**, helping organizations quickly confirm remediation efforts.

---

## ➕ Summary of Benefits

| Feature                | Benefit                                                               |
| ---------------------- | --------------------------------------------------------------------- |
| Multi-Sample Sort      | Reduces LLM randomness and increases stability                        |
| Recursive Refinement   | Focuses compute power on high-ranking items for better final ordering |
| Linear Scaling         | Efficient even for large input sets (O(N) complexity)                 |
| Cybersecurity Use Case | Enhanced vulnerability triage; reduced manual review burden           |
| Open-source & Flexible | Easy to integrate with LLM pipelines via CLI                          |

---

