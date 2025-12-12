## Memory Lifecycle Control Under a Fixed Budget

Let the input sequence be processed in steps t = 1..T producing a context vector c_t ∈ R^d (e.g., CLS state).
We maintain an external memory with a fixed number of slots N. Memory at step t is:

M_t = {(m_i^t, a_i^t, u_i^t)}_{i=1..N}

where m_i^t ∈ R^D is the slot content, a_i^t is an age signal, and u_i^t is a usage signal.

A neural controller produces lifecycle actions per slot:
p_i^t = softmax(gθ(m_i^t, c_t, a_i^t, u_i^t)) ∈ Δ^3
corresponding to retain, update, forget probabilities.

A budgeted allocator selects at most K operations per step from N slots (and an optional write action),
ensuring bounded memory management cost and preventing uncontrolled overwriting.

State transition:
M_{t+1} = Update(M_t, x_t; θ)
with slot-wise transitions implementing:
- retain: preserve slot content and increment age
- update: refine slot content conditioned on c_t
- forget: clear slot content (freeing capacity)
- write: insert new information into a freed or low-utility slot

Objective:
minimize task loss while learning stable lifecycle behavior under fixed budget constraints,
measured by task performance and memory metrics (utilization, churn, retention duration, forgetting rate).
