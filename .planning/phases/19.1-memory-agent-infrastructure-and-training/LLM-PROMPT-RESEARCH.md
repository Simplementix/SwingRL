# LLM Prompt Engineering Research: Structured Output & Hallucination Reduction

Research findings for SwingRL Phase 19.1 consolidation prompts. All recommendations are actionable and specific to our stack (NVIDIA NIM with Kimi K2.5, Ollama with Qwen3:14b fallback).

---

## 1. Structured Output Enforcement

### 1A. NVIDIA NIM: Use `guided_json` (NOT `response_format`)

NVIDIA NIM supports OpenAI-compatible structured output, but has its own preferred mechanism:

- **`response_format: {"type": "json_schema", ...}`** -- Works, but is the slower path. NIM translates this internally.
- **`guided_json` via `nvext`** -- NVIDIA's recommended approach. Uses the xgrammar backend, which is the fastest structured generation option. Pass it through `extra_body`:

```python
extra_body={"nvext": {"guided_json": your_json_schema}}
```

- **Avoid `{"type": "json_object"}`** -- This only guarantees valid JSON, not schema conformance. NVIDIA explicitly recommends `json_schema` or `guided_json` over plain `json_object` mode.
- **Fallback concern**: If xgrammar cannot handle your schema (deeply nested, recursive), NIM falls back to the `outlines` backend, which has significant first-inference latency. Keep schemas flat.

**Recommendation for our code**: Use `guided_json` via `nvext` for NVIDIA NIM. For OpenAI-compatible fallback path (if we ever switch providers), also support `response_format: {"type": "json_schema"}`. Abstract this behind a provider-aware wrapper.

### 1B. Ollama (Qwen3:14b): Use `format` Parameter with JSON Schema

Since Ollama v0.5, you can pass a full JSON schema (not just `"json"`) to the `format` parameter:

```python
format=your_json_schema_dict  # Full JSON schema, not just "json"
```

- Ollama generates a grammar from the schema and constrains decoding at the token level via llama.cpp.
- **Critical limitation**: Thinking mode (reasoning mode) is incompatible with structured output in Ollama. Since Qwen3 has a thinking mode, you must ensure it is disabled when requesting structured output (`/no_think` prefix or model parameter).
- **Validation gap**: Ollama does NOT validate the full response against the schema. If the model stops mid-generation (hits token limit), you get truncated invalid JSON despite grammar constraints. Always validate the parsed JSON yourself.
- **Keep schemas simple**: Deeply nested or recursive schemas cause issues. Flat with one level of nesting is safest.

**Recommendation**: Always set `temperature=0` for structured output calls to Ollama. Always wrap the response in a try/except JSON parse. Always disable thinking mode.

### 1C. Enforcing JSON Arrays with Specific Fields

To get a JSON array (e.g., list of patterns), wrap it in a root object:

```json
{
  "type": "object",
  "properties": {
    "patterns": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "name": {"type": "string"},
          "category": {"type": "string", "enum": ["market_regime", "risk", "alpha", "execution"]},
          "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0}
        },
        "required": ["name", "category", "confidence"]
      }
    }
  },
  "required": ["patterns"]
}
```

JSON Schema's `enum` keyword is the strongest way to constrain categorical values. Both NIM's guided_json and Ollama's format parameter enforce `enum` at the grammar/token level -- the model literally cannot generate a value outside the enum list.

---

## 2. Hallucination Reduction for Consolidation/Summarization

### 2A. Grounding Techniques

The most effective prompt-level grounding techniques for our consolidation task:

1. **Explicit source-only instruction**: Add to system prompt:
   ```
   You are a data analyst. You MUST only reference data, metrics, and facts that appear explicitly in the input below. If a claim cannot be directly supported by a specific number or fact in the input, do not include it. Never infer trends that are not directly evidenced by the data.
   ```

2. **Quote-and-cite pattern**: Require the model to cite the specific input data that supports each claim:
   ```
   For each pattern you identify, include a "evidence" field that quotes the specific metrics from the input that support this pattern.
   ```
   This forces the model to ground every output claim in input data. If it cannot find supporting evidence, the `evidence` field will be weak/vague, which is a detectable signal.

3. **Input delimiting**: Clearly separate the data from instructions using XML-style tags or triple-backtick fences:
   ```
   <training_data>
   {the actual fold results here}
   </training_data>

   Analyze ONLY the data between the <training_data> tags.
   ```

4. **"I don't know" permission**: Explicitly tell the model it can say "insufficient data" or return fewer patterns than requested:
   ```
   If the data does not support a clear pattern, return fewer items rather than speculating. An empty patterns array is acceptable.
   ```

### 2B. Few-Shot Example Framing (Critical)

**The problem**: If you provide few-shot examples with specific values (e.g., "PPO outperforms SAC in volatile regimes"), the model may parrot those specific claims regardless of the actual input data.

**The fix**: Frame examples explicitly as format demonstrations with synthetic/placeholder data:

```
The following is a FORMAT EXAMPLE ONLY. The data values are fictional placeholders.
Do NOT treat any claims in this example as factual. Your output must be based
solely on the actual input data provided after this example.

EXAMPLE FORMAT:
{
  "patterns": [
    {
      "name": "PLACEHOLDER_PATTERN_NAME",
      "category": "market_regime",
      "confidence": 0.42,
      "evidence": "PLACEHOLDER: cite specific metrics from input",
      "description": "PLACEHOLDER: describe what the data shows"
    }
  ]
}
END OF FORMAT EXAMPLE.
```

Key principles:
- Use obviously fake placeholder values ("PLACEHOLDER_PATTERN_NAME") so the model cannot confuse them with real data.
- Explicitly label the boundary between the example and the real task.
- Use a confidence value in the example that is NOT 0.8 or 0.9 (to avoid anchoring -- see Section 3B).

### 2C. Chain-of-Thought vs Direct Output

**Recent 2025 research finding** (Wharton / ICLR 2025): CoT primarily helps on math and symbolic reasoning tasks. For analytical/summarization tasks, CoT provides much smaller gains and can actually introduce variability that hurts accuracy on straightforward cases. CoT also costs 35-600% more tokens.

**For our consolidation task**:
- The task is analytical extraction (finding patterns in structured numerical data), not mathematical reasoning.
- CoT is likely to INCREASE hallucination risk because the "reasoning" steps give the model more opportunities to generate plausible-but-unsupported intermediate claims.
- **Recommendation**: Use direct output mode. Ask for the structured JSON directly, not "think step by step then produce JSON."
- If you want verification, use a separate validation pass (generate, then ask a second call to verify claims against input) rather than inline CoT.

### 2D. Temperature and Sampling Parameters

**Consensus from 2024-2025 research**:

| Parameter | Recommended Value | Rationale |
|-----------|------------------|-----------|
| Temperature | 0.0-0.2 | Factual/analytical tasks need determinism. 0.0 for schema-constrained output. |
| Top-p | 0.9-0.95 | Standard. Lower values (0.8) if you see creative drift. |
| Top-k | 40-50 | Only relevant if provider supports it. Limits token candidates. |
| Max tokens | Set explicitly | Prevent truncation. Calculate based on expected output size + 50% buffer. |
| Frequency penalty | 0.0 | Do NOT penalize repetition for structured output -- pattern names may legitimately repeat terms. |

**Key insight**: With grammar-constrained output (guided_json / Ollama format), temperature matters less for schema compliance (the grammar enforces it), but still matters for the CONTENT of string fields. Use temperature=0.1 (not exactly 0.0) for slight variation between runs if you want to detect unstable patterns via multi-pass comparison.

---

## 3. Pattern Extraction Prompts

### 3A. Preventing Invented Patterns

The core risk: LLMs are excellent at generating plausible-sounding patterns from any data, even random noise. Techniques to combat this:

1. **Require quantitative evidence**: Every pattern must cite specific numbers from the input:
   ```
   Each pattern MUST include at least two specific metric values from the input data
   that support it. Example: "mean_reward improved from -0.23 to 0.41 across folds 3-5"
   ```

2. **Minimum support threshold**: Define what counts as a "pattern" vs. noise:
   ```
   A pattern requires consistent evidence across at least 3 folds or 2 algorithms.
   A single anomalous fold is an observation, not a pattern.
   ```

3. **Constrain pattern count**: Do NOT say "find all patterns." Say:
   ```
   Identify between 0 and 5 patterns. Prefer fewer, well-supported patterns over many weak ones.
   Return an empty array if the data does not support clear patterns.
   ```

4. **Adversarial framing**: Instruct the model to be skeptical:
   ```
   Act as a skeptical peer reviewer. For each candidate pattern, ask: "Could this be
   explained by random variance across folds?" If yes, do not include it.
   ```

### 3B. Confidence Calibration

**The problem**: LLMs almost always output confidence scores clustered in 0.75-0.95, regardless of actual certainty. Research confirms Expected Calibration Errors of 0.10-0.43 in LLM-generated confidence scores.

**Techniques that actually work**:

1. **Anchored scale with explicit definitions** (most effective for prompt-only approach):
   ```
   Confidence scoring guide:
   - 0.9-1.0: Pattern is mathematically certain from the data (e.g., "Algorithm X had highest reward in ALL folds")
   - 0.7-0.89: Strong pattern with 1-2 exceptions (e.g., "Algorithm X outperformed in 4 of 5 folds")
   - 0.5-0.69: Moderate pattern, roughly half the evidence supports it
   - 0.3-0.49: Weak pattern, slight trend but substantial counter-evidence
   - 0.0-0.29: Very weak, barely distinguishable from noise

   Most patterns from noisy financial data should fall in the 0.4-0.7 range.
   A confidence above 0.85 requires overwhelming, exception-free evidence.
   ```

2. **Calibration via base rate instruction**:
   ```
   In our experience, only ~10% of detected patterns have confidence above 0.8.
   Calibrate your scores accordingly.
   ```

3. **Multi-pass consistency check** (post-processing, not prompt): Run the same prompt 3-5 times with temperature=0.1. Patterns that appear in all runs with similar confidence are likely real. Patterns that appear sporadically are likely hallucinated. This is the most reliable approach but costs 3-5x in API calls.

4. **Avoid anchoring in examples**: If your few-shot example shows confidence=0.85, the model will anchor near that value. Use intentionally varied confidence values in examples (e.g., 0.42, 0.67) and include at least one low-confidence example.

### 3C. Category/Enum Assignment

To ensure the model picks from a defined enum:

1. **Schema-level enforcement** (strongest): Use `"enum": [...]` in your JSON schema. With guided_json/format, this is enforced at the token level. The model cannot output a category not in the list.

2. **Prompt-level reinforcement** (belt and suspenders):
   ```
   category MUST be one of: "market_regime", "risk", "alpha", "execution", "data_quality".
   If a pattern does not cleanly fit any category, use the closest match.
   Do NOT create new categories.
   ```

3. **Category definitions** (reduces miscategorization):
   ```
   Category definitions:
   - market_regime: Patterns related to how algorithms perform under different market conditions
   - risk: Patterns related to drawdowns, volatility, risk-adjusted returns
   - alpha: Patterns related to absolute returns, reward signals, profitability
   - execution: Patterns related to training stability, convergence, computational performance
   - data_quality: Patterns related to data gaps, feature quality, input anomalies
   ```

---

## 4. Prompt Structure Best Practices

### 4A. System Prompt vs User Prompt Separation

**What goes in the system prompt** (persistent behavioral constraints):
- Role definition ("You are a quantitative analyst...")
- Output format rules and JSON schema description
- Grounding rules ("Only cite data from the input")
- Confidence calibration scale
- Enum definitions and category descriptions
- "Do not" rules (negative instructions)

**What goes in the user prompt** (per-request variable content):
- The actual training data to analyze
- The specific question/task for this request
- Few-shot format examples (with the "FORMAT ONLY" framing)

**Rationale**: System prompts are treated with higher priority by most models. Putting behavioral constraints there makes them harder to override. The user prompt should contain only the variable parts that change per request.

### 4B. Few-Shot Examples as Format Templates

Recommended structure:

```
SYSTEM:
[Role + behavioral rules + schema + calibration scale + enum definitions]

USER:
Below is training run data to analyze.

--- FORMAT TEMPLATE (fictional data, for output structure only) ---
Input: [2-3 lines of obviously fake data]
Output:
{"patterns": [{"name": "EXAMPLE_ONLY", "category": "risk", "confidence": 0.55, "evidence": "fictional metric: 0.xx", "description": "This is a format example only"}]}
--- END FORMAT TEMPLATE ---

--- ACTUAL DATA ---
<training_data>
{real data here}
</training_data>

Analyze the data above. Return a JSON object matching the format template.
```

### 4C. Negative Instructions Effectiveness

Research findings on "Do NOT..." instructions:

- **They work, but are weaker than positive alternatives**. Models are better at following "Do X" than "Don't do Y" because the negative instruction still activates the concept it's trying to suppress.
- **Most effective when paired with the positive alternative**:
  - Weak: "Do NOT invent patterns not supported by data."
  - Strong: "Only include patterns with direct numerical evidence from the input. If in doubt, omit the pattern."
- **Effective for specific, concrete prohibitions**:
  - Weak: "Do NOT hallucinate." (too vague)
  - Strong: "Do NOT reference any ticker symbols, dates, or metric values that do not appear in the input data."
- **Place negative instructions in the system prompt** where they carry more weight, not buried in the user prompt.
- **Limit to 3-5 negative instructions** max. Long lists of "do not" rules cause the model to fixate on them and sometimes violate them more.

### 4D. Complete Recommended Prompt Architecture

```
SYSTEM PROMPT:
1. Role: "You are a quantitative analyst reviewing algorithmic trading results."
2. Task framing: "Your job is to identify statistically meaningful patterns..."
3. Grounding rule: "You MUST only reference metrics that appear in the input data."
4. Output schema description (brief)
5. Confidence calibration scale (from 3B above)
6. Category enum definitions (from 3C above)
7. Key constraints (3-4 negative instructions, each paired with positive alternative)

USER PROMPT:
1. Format template (clearly labeled as fictional)
2. Data delimiter: <training_data>...</training_data>
3. Specific task instruction
4. Reminder: "Return valid JSON matching the schema. 0-5 patterns. Empty array is acceptable."
```

---

## Summary of Top Actionable Recommendations

1. **Use `guided_json` via nvext for NVIDIA NIM**, `format` with full JSON schema for Ollama. Both enforce enum at token level.
2. **Disable Qwen3 thinking mode** when requesting structured output from Ollama.
3. **Temperature 0.0-0.1** for all consolidation calls. Set max_tokens explicitly.
4. **Skip CoT** for consolidation -- direct output reduces hallucination and token cost.
5. **Require quantitative evidence** in each pattern (specific numbers from input, minimum 2).
6. **Frame few-shot examples with PLACEHOLDER values** and explicit "FORMAT ONLY" labels.
7. **Use anchored confidence scale** with base rate instruction ("most patterns should be 0.4-0.7").
8. **Place all behavioral rules in system prompt**, all variable data in user prompt.
9. **Pair every negative instruction with its positive alternative**.
10. **Validate JSON client-side** -- neither NIM nor Ollama guarantees complete valid JSON on truncation.
11. **Consider multi-pass consistency** (3 runs at temperature=0.1) for high-stakes consolidation to filter hallucinated patterns.

---

## Sources

- [OpenAI Structured Outputs Guide](https://platform.openai.com/docs/guides/structured-outputs)
- [OpenAI Introducing Structured Outputs](https://openai.com/index/introducing-structured-outputs-in-the-api/)
- [NVIDIA NIM Structured Generation (latest)](https://docs.nvidia.com/nim/large-language-models/latest/structured-generation.html)
- [NVIDIA NIM Structured Generation (v1.2)](https://docs.nvidia.com/nim/large-language-models/1.2.0/structured-generation.html)
- [Ollama Structured Outputs Docs](https://docs.ollama.com/capabilities/structured-outputs)
- [Ollama Blog: Structured Outputs](https://ollama.com/blog/structured-outputs)
- [Constraining LLMs with Structured Output: Ollama, Qwen3](https://medium.com/@rosgluk/constraining-llms-with-structured-output-ollama-qwen3-python-or-go-2f56ff41d720)
- [Ollama Thinking Mode + Structured Output Issue #10929](https://github.com/ollama/ollama/issues/10929)
- [Ollama Structured Outputs for Reasoning Models Issue #10538](https://github.com/ollama/ollama/issues/10538)
- [Structuring Enums for LLM Results with Instructor](https://ohmeow.com/posts/2024-07-06-llms-and-enums.html)
- [LiteLLM JSON Mode Docs](https://docs.litellm.ai/docs/completion/json_mode)
- [vLLM Structured Outputs](https://docs.vllm.ai/en/latest/features/structured_outputs/)
- [Hallucination Taxonomy Survey (MDPI 2025)](https://www.mdpi.com/2673-2688/6/10/260)
- [Hallucination Mitigation Survey (arxiv 2510.24476)](https://arxiv.org/html/2510.24476v1)
- [Lakera Guide to LLM Hallucinations](https://www.lakera.ai/blog/guide-to-hallucinations-in-large-language-models)
- [Epistemic Stability for Industrial LLM Hallucination Reduction (arxiv 2603.10047)](https://arxiv.org/abs/2603.10047)
- [Prompt Engineering Patterns that Reduce Hallucinations (ResearchGate)](https://www.researchgate.net/publication/394431721_Prompt_Engineering_Patterns_that_Reduce_Hallucinations_in_Large_Language_Models)
- [7 Prompt Engineering Tricks to Mitigate Hallucinations](https://machinelearningmastery.com/7-prompt-engineering-tricks-to-mitigate-hallucinations-in-llms/)
- [5 Methods for Calibrating LLM Confidence Scores](https://latitude.so/blog/5-methods-for-calibrating-llm-confidence-scores)
- [On Verbalized Confidence Scores for LLMs (arxiv 2412.14737)](https://arxiv.org/pdf/2412.14737)
- [Wharton: Decreasing Value of Chain of Thought](https://gail.wharton.upenn.edu/research-and-insights/tech-report-chain-of-thought/)
- [To CoT or not to CoT? (ICLR 2025)](https://openreview.net/forum?id=w6nlcS8Kkn)
- [System Prompts vs User Prompts Design Patterns](https://tetrate.io/learn/ai/system-prompts-vs-user-prompts)
- [Palantir LLM Prompt Engineering Best Practices](https://www.palantir.com/docs/foundry/aip/best-practices-prompt-engineering)
- [Lakera Prompt Engineering Guide 2026](https://www.lakera.ai/blog/prompt-engineering-guide)
- [LLM Temperature (IBM)](https://www.ibm.com/think/topics/llm-temperature)
- [Temperature Effect on Problem Solving (arxiv 2402.05201)](https://arxiv.org/html/2402.05201v1)
- [Prompt Engineering Guide: LLM Settings](https://www.promptingguide.ai/introduction/settings)
