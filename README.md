# eCommerce-AI-Agent-
eCommerce AI Agent for Customer Service

This is an example implementation of the simulation for $\tau$-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Retail eCommerce customer service (https://arxiv.org/pdf/2406.12045), using AWS Strands Agent SDK (https://strandsagents.com/). 

## 1. Agent Framework

An LLM agent system is defined by the tuple **(S, A, O, T, π, R, I)** where:

1. **S = S_world ⊗ S_history** state decomposes into world state (databases, files, physical environment) and interaction history (all prior actions and observations)
2. **A = A_tool ∪ A_msg** actions are either tool calls that modify the environment or communicative acts (messages)
3. **O = O_tool ∪ O_msg** observations are either tool return values or received messages
4. **T: S × A → S × O** transition function specifying how actions change the world state and produce observations
5. **π: S_history → Δ(A)** the policy, implemented by the LLM, mapping observable history to a distribution over actions
6. **R: S_world → [0,1]** evaluation function, defined on terminal world state
7. **I ∈ S_history** initial instruction (system prompt, task description, policy documents)

In $\tau$-bench, the Retail LLM agent interacts with a simulated user and hence, the environment is interactive. The world state is the Retail Database that consist of the Product information, and User information and their order history. 

| Symbol | τ-bench Retail Implementation |
|---|---|
| **S_world** | Retail database: 500 users, 50 products, 1000 orders (mutable) |
| **S_history** | `agent.messages` — full conversation + tool call history |
| **A_tool** | 16 tools: `find_user_id_by_email`, `cancel_pending_order`, etc. |
| **A_msg** | Messages sent to user (asking questions, confirming actions) |
| **O_tool** | Tool return values (order details JSON, error messages) |
| **O_msg** | User simulator responses |
| **T** | Tool execution against `db` — modifies orders, balances, etc. |
| **π** | Claude Sonnet on Bedrock (frozen, stochastic with temp > 0) |
| **R** | Action matching against expected actions in `tasks.json` |
| **I** | `policy.md` as system prompt |

### 1.2 The Agentic Loop (per turn)

```
1. User says something (O_msg) → appended to h_t
2. Agent sees full h_t → generates action (tool call or message)
3. If tool call: T executes it, returns O_tool → appended to h_t → back to 2
4. If message: sent to user simulator → user responds → back to 1
5. Repeat until user says ###STOP### or max_turns reached
6. Evaluate: R(final DB state, tool trace) vs expected actions
```
---

## 2. Retail Database

A single JSON file acting as the entire retail database, with three tables:

| Table | Count | What's in it |
|---|---|---|
| `products` | 50 | Product types with variants (item IDs, options, prices, availability) |
| `users` | 500 | Customer profiles (name, email, address, payment methods, order history) |
| `orders` | 1000 | Orders (status, items, fulfillment tracking, payment history) |

The agent's tools query and modify this in-memory dict during each task run. It gets reset (fresh deep copy from `ORIGINAL_DB`) before each new task to prevent state leakage between tasks.

Order statuses: `pending`, `processed`, `delivered`, `cancelled`
Payment methods: `gift_card`, `credit_card`, `paypal`

---

## 3. What makes τ-bench hard 

### It's a partially observable Markov Decision Process (POMDP)

The agent cannot see:
1. The user's actual goal (only inferred from conversation)
2. Which items the user prefers (revealed progressively)
3. Whether the user will confirm or change their mind

### Dual-source stochasticity (why pass^k matters)

Two LLMs generating tokens stochastically:
1. **Agent LLM** might ask different questions, call tools in different order
2. **User simulator LLM** might phrase responses differently each run

Same task, same starting state → potentially different conversation → different outcome.

### Policy compliance under pressure

The agent must simultaneously:
1. Be conversational and helpful (UX quality)
2. Follow strict policy rules (never cancel delivered orders, must confirm before acting)
3. Gather information progressively (can't ask user to dump everything at once)
4. Handle ambiguity (user says "my order" — which one?)

---

## 4. Evaluation metrics 

### Primary: Action Matching (τ-bench's core metric)

Each task defines `evaluation_criteria.actions` — the expected sequence of tool calls with
exact arguments. The agent passes only if it calls the right tools with the right arguments.

```python
# From tasks.json, task 0:
"actions": [
    {"name": "find_user_id_by_name_zip", "arguments": {"first_name": "Yusuf", "last_name": "Rossi", "zip": "19122"}},
    {"name": "get_order_details", "arguments": {"order_id": "#W2378156"}},
    {"name": "get_product_details", "arguments": {"product_id": "1656367028"}},
    {"name": "exchange_delivered_order_items", "arguments": {...}},
]
```

### Reliability: pass^k

```
pass^1 = fraction of tasks the agent gets right on a single attempt
pass^4 = fraction of tasks the agent gets right on ALL 4 attempts
```

### Additional metrics to track:

| Metric | What to observe |
|---|---|
| **Action recall** | What fraction of expected tools were called? |
| **Action precision** | How many unnecessary tool calls? |
| **Turn count** | How many conversation turns to complete? (efficiency) |
| **Token usage** | Total input + output tokens (cost proxy) |
| **Elapsed time** | Wall-clock seconds per task |
| **Policy violations** | Did agent call a write tool without confirming first? |
| **Hallucination** | Did agent claim facts not in tool outputs? |

---
## 5. Key observations to record

After each experiment, note:

| What to record | Why |
|---|---|
| pass^1 and pass^k | Core benchmark numbers |
| Which tasks consistently fail | Identifies systematic weaknesses |
| Which tasks are inconsistent (pass sometimes) | These drive pass^k down |
| Average tool calls per task | Efficiency |
| Average turns per task | Conversation length |
| Total cost (tokens) | Practical deployment consideration |
| Common failure modes | Guides improvement |
| User simulator errors | Distinguish agent failure from sim failure |

---

## 6. Things to experiment with

1. **Baseline Performance** What's the baseline pass^1?
2. **Reliability (pass^k)** How much does pass^k drop from pass^1? Which tasks are inconsistent?
3. **Different Models** How do pass^1 and pass^k compare across models?
4. **Agent/User Prompts** How do pass^1 and pass^k changes with change in Agent/User system prompts?
5. **Error Analysis** Which tasks have failures? Where did it fail? Is it a reasoning failure or communication failure?
6. **Scaffolding improvements** Do scaffolding improvmenets such as Retry-on-error or ReAct step for Agent improve the metrics?
7. **Harness modifications** Is the evaluation harness too strict? 

---





