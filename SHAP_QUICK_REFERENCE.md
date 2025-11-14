# SHAP Quick Reference

## Function Overview

### Core Functions

#### 1. `_make_shap_explainer(domain_key, background_data=None)`
Creates and caches SHAP KernelExplainer

```python
# Returns: (explainer, predict_fn, labels, preproc)
explainer, pred_fn, labels, preproc = _make_shap_explainer("movies")
```

**Parameters:**
- `domain_key`: "books", "movies", or "games"
- `background_data`: Optional pre-encoded background samples (default: 10 neutral texts)

**Features:**
- Auto-creates background data if not provided
- Returns cached resources to avoid recomputation
- Integrates with existing model loading

---

#### 2. `_compute_shap_values_for_text(domain_key, user_text, target_idx)`
Computes SHAP values for a given text and class

```python
# Returns: Tuple[np.ndarray, List[str]] or None
shap_vals, tokens = _compute_shap_values_for_text("movies", "sci-fi text here", 0)
```

**Parameters:**
- `domain_key`: Domain identifier
- `user_text`: Text to explain
- `target_idx`: Class index to explain (0-indexed)

**Output:**
- `shap_vals`: Array of SHAP values [shape: max_length]
- `tokens`: List of tokenized words
- Returns `None` if computation fails

**Raises:**
- `ImportError` if SHAP not installed

---

#### 3. `_extract_shap_features(shap_values, tokens, num_features=10)`
Extracts top-N most important tokens

```python
# Returns: pd.DataFrame
top_features = _extract_shap_features(shap_vals, tokens, num_features=10)
```

**Parameters:**
- `shap_values`: SHAP values array
- `tokens`: List of tokens
- `num_features`: Number of top features to return (default: 10)

**Output DataFrame:**
```
      token  shap_value
0   sci-fi         0.45
1 thrilling         0.32
2     action         0.28
```

**Columns:**
- `token`: Feature (word/token)
- `shap_value`: SHAP value (positive = supports, negative = opposes)

---

## UI Integration Points

### User Controls
```python
# Lines ~1000-1001
shap_enabled = st.checkbox("Run SHAP explanation", value=False)
```

### Execution Block
```python
# Lines ~1065-1110
if shap_enabled:
    # Compute SHAP values
    # Extract top features
    # Display results
```

---

## Data Flow Diagram

```
User Selects SHAP
      │
      ▼
[Check SHAP Available]
      │
   ┌──┴──┐
   │     │
   NO    YES
   │     │
   │     ▼
   │  [Get predict_fn, labels]
   │     │
   │     ▼
   │  [Get target_idx]
   │     │
   │     ▼
   │  [_compute_shap_values_for_text]
   │     ├─ Create explainer
   │     ├─ Encode text
   │     ├─ Call explainer.shap_values()
   │     ├─ Extract target class values
   │     └─ Tokenize for alignment
   │     │
   │     ▼
   │  [_extract_shap_features]
   │     ├─ Get top 10 by absolute value
   │     └─ Return DataFrame
   │     │
   │     ▼
   │  [Display Results]
   │     ├─ DataFrame table
   │     └─ Bar chart
   │
   └──► [Error Handling]
        ├─ Log exception
        └─ Show user message
```

---

## Key Parameters & Constants

```python
# Hardcoded
num_features = 10              # Top features to show
background_samples = 10        # Size of background dataset
num_tokens = min(len(tokens), len(shap_values))  # Align padding

# From preprocessor.json
max_length = 256               # Sequence padding length
lowercase = True               # Token normalization
oov_token_id = 1              # Unknown token ID
```

---

## Output Formats

### Table Format (Displayed to User)
```
      token  shap_value
0   sci-fi    0.451234
1 thrilling   0.328901
2     action   0.287654
3    amazing   0.156789
...
```

### Bar Chart Format
- X-axis: Token names
- Y-axis: SHAP values (magnitude)
- Positive bars: Support prediction
- Negative bars: Oppose prediction

### Raw SHAP Output
```python
shap_vals: np.ndarray [shape: (max_length,)]
tokens: List[str] (length varies, typically 10-100)
```

---

## Comparison: LIME vs SHAP

### LIME
```
Input: "text here"
  │
  ├─ Generate 500 perturbed variations
  ├─ Get predictions for each
  └─ Fit local linear model
  
Output: Feature weights
```

### SHAP
```
Input: "text here"
  │
  ├─ Load background dataset
  ├─ Create KernelExplainer
  ├─ Compute Shapley values
  └─ Extract features
  
Output: Shapley values
```

### Performance
| Metric | LIME | SHAP |
|--------|------|------|
| Time | ~2-3 sec | ~5-10 sec |
| Memory | ~20 MB | ~30 MB |
| Theory | Local approx | Game theory |
| Consistency | Approximate | Exact |

---

## Error Handling

### Error Scenarios

**SHAP not installed**
```python
if shap is None:
    st.info("SHAP not installed. Please add 'shap'...")
    # User sees friendly message
```

**Missing model artifacts**
```python
try:
    _compute_shap_values_for_text(...)
except FileNotFoundError as e:
    logger.warning("Artifacts missing for SHAP: %s", e)
    st.info(str(e))
```

**Computation fails**
```python
shap_result = _compute_shap_values_for_text(...)
if shap_result is None:
    st.warning("SHAP computation failed.")
```

**Empty features**
```python
if not shap_df.empty:
    st.dataframe(shap_df)
else:
    st.warning("No SHAP features extracted.")
```

---

## Caching Strategy

| Component | Cache Type | Scope |
|-----------|-----------|-------|
| Model bundle | `@st.cache_resource` | Session-wide |
| SHAP explainer | Manual (in function) | Per-domain |
| SHAP computation | None | Per-request |

**Cache Miss Triggers:**
- New domain selected
- Streamlit app restarted
- Cache cleared manually

---

## Integration with Existing Code

### Reused Components
```python
_load_bundle_for_explain()  # Model + vocab loading
_make_predict_fn()          # Prediction wrapper
_encode_texts()             # Tokenization + encoding
_tokenize()                 # Text tokenization
```

### No Duplication
- ✅ Single model instance
- ✅ Shared vocabulary
- ✅ Same encoding pipeline
- ✅ Common error patterns

---

## Usage Examples

### Basic SHAP Explanation
```python
# In app.py predict block
target_idx = labels.index(pred_label)
shap_result = _compute_shap_values_for_text(domain, user_text, target_idx)

if shap_result:
    shap_vals, tokens = shap_result
    shap_df = _extract_shap_features(shap_vals, tokens, num_features=10)
    st.dataframe(shap_df)
```

### Custom Background Data
```python
# For more specific explanations
custom_texts = ["a", "the", "and", ...]  # Your background texts
custom_bg = _encode_texts(custom_texts, preproc)
explainer, _, _, _ = _make_shap_explainer(domain, background_data=custom_bg)
```

### Adjusting Feature Count
```python
# Show top 15 features instead of 10
shap_df = _extract_shap_features(shap_vals, tokens, num_features=15)
```

---

## Debugging Tips

### Check if SHAP is working
```python
import shap
print(shap.__version__)  # Should print version >= 0.44.1
```

### Verify explainer creation
```python
try:
    explainer, _, _, _ = _make_shap_explainer("movies")
    print("Explainer created successfully")
except Exception as e:
    print(f"Error: {e}")
```

### Check SHAP value output
```python
shap_vals, tokens = _compute_shap_values_for_text("movies", "test text", 0)
print(f"SHAP shape: {shap_vals.shape}")
print(f"Tokens: {len(tokens)}")
print(f"Top SHAP values: {np.sort(np.abs(shap_vals))[-10:]}")
```

### Verify alignment
```python
print(f"SHAP values: {len(shap_vals)}")
print(f"Tokens: {len(tokens)}")
print(f"Max length from preproc: {max_length}")
```

---

## Logging Output

### Log Levels Used

**INFO**
```
Initializing SHAP explainer for domain=movies
Target index for SHAP: 2 (Sci-Fi)
SHAP computation successful: 25 tokens
```

**DEBUG**
```
Computing SHAP values for user_text (len=156)
SHAP values computed: shape=(256,)
User text tokenized into 25 tokens
Extracted 10 top SHAP features
```

**WARNING**
```
SHAP not installed or failed to import
Artifacts missing for SHAP: Missing model.ts/labels.txt/preprocessor.json
```

**EXCEPTION**
```
Failed to create SHAP explainer: [exception details]
Failed to compute SHAP values: [exception details]
```

---

## File Dependencies

### Input Files (Required)
```
artifacts/{domain}/
├── model.ts              # TorchScript model
├── labels.txt            # Class labels
└── preprocessor.json     # Vocabulary & config
```

### Output (User Facing)
```
DataFrame Columns:
├── token       (str)     # Feature token
└── shap_value  (float)   # SHAP value
```

---

## Performance Tips

### Optimize SHAP Computation
1. **Use cached model**: `@st.cache_resource` handles this
2. **Minimal background**: 10 samples is minimum
3. **Smaller max_length**: Faster padding/encoding
4. **GPU inference**: Set device="cuda" (if available)

### Optimize Display
1. **Lazy loading**: Only compute if user enabled
2. **Cache results**: Store for identical inputs
3. **Batch processing**: Multiple texts at once

---

## Common Patterns

### Pattern 1: Basic Explanation
```python
if shap_enabled and shap is not None:
    shap_result = _compute_shap_values_for_text(domain, user_text, target_idx)
    if shap_result is not None:
        shap_vals, tokens = shap_result
        shap_df = _extract_shap_features(shap_vals, tokens)
        st.dataframe(shap_df)
```

### Pattern 2: Multi-class Comparison
```python
for class_idx, class_name in enumerate(labels):
    shap_result = _compute_shap_values_for_text(domain, user_text, class_idx)
    if shap_result:
        shap_vals, tokens = shap_result
        # Compare across classes
```

### Pattern 3: Batch Explanations
```python
for text in text_batch:
    shap_result = _compute_shap_values_for_text(domain, text, target_idx)
    # Accumulate results
```

---

## Related Files

- **app.py** - Main Streamlit app (this implementation)
- **SHAP_IMPLEMENTATION_GUIDE.md** - Detailed technical guide
- **LIME_QUICK_REFERENCE.md** - LIME comparison
- **LIME_IMPLEMENTATION_ANALYSIS.md** - LIME analysis
- **inference.py** - `predict_with_weights()` function
- **requirements.txt** - Package dependencies

---

## Summary Table

| Feature | Value |
|---------|-------|
| **Method** | KernelExplainer (SHAP) |
| **Input** | Text string |
| **Output** | Token importance scores |
| **Time** | ~3-8 seconds |
| **Memory** | ~30 MB |
| **Default Features** | 10 |
| **Background Samples** | 10 |
| **Cached** | Model & explainer |
| **Optional** | Yes (disabled by default) |
| **Complementary to** | LIME |

