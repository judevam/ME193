# V-fit warm-up: explanations, equations, and Excel recipe

You do the build; this is the reference to build from. Equations are written so you can paste them into Notion with `/equation` (the `$$ ... $$` parts).

## 0. What the assignment wants, and where you stand

| Piece | Status |
|---|---|
| Python fit of the V data, plot of data + fit, plot of loss vs epoch | **`fit_v.py` already does this** (and `fit_v.png` is its output). Check it matches what you want to show. |
| Math: gradient descent for a **single neuron** | Section 2 |
| Math: gradient descent for a **ReLU layer** | Sections 3-4 |
| Excel with **5 iterations on m and b** | Section 5 |
| Short explanation of what the plots show | Section 6 |

## 1. The data, in plain words (your numbers)

`annacode-v.tsv`: 120 points, x from 21 to 100, y from about -2 to 78.
The points follow a V: roughly **y = 2 |x - 60|**. The bottom is near x = 60, y = 0; the right arm rises with slope about +2, the left arm falls with slope about -2; the scatter is about 1.8.

Useful contrast for the write-up: the best **single straight line** through all 120 points is m = -0.09, b = 41 with an RMS error of 23.5. That is terrible, because a line cannot bend. A line fit to only the **right arm** (x > 65) gets m = 1.98, b = -118, RMS 2.1. So: one linear neuron cannot make a V, but one ReLU can make one arm, and two ReLUs can make the whole V.

## 2. A single neuron: y-hat = m x + b

Model (one input x, weight m, bias b):

$$\hat y_i = m\,x_i + b$$

Loss, the mean squared error over N points:

$$L(m,b)=\frac1N\sum_{i=1}^{N}\left(\hat y_i-y_i\right)^2$$

Let the error of point i be $e_i=\hat y_i-y_i$. Use the chain rule: $L=\frac1N\sum e_i^2$, and $\partial e_i/\partial m = x_i$, $\partial e_i/\partial b = 1$. So

$$\frac{\partial L}{\partial m}=\frac2N\sum_i e_i\,x_i\qquad\qquad \frac{\partial L}{\partial b}=\frac2N\sum_i e_i$$

Gradient descent takes a small step downhill, with learning rate $\eta$ (eta):

$$m\leftarrow m-\eta\,\frac{\partial L}{\partial m}\qquad\qquad b\leftarrow b-\eta\,\frac{\partial L}{\partial b}$$

Intuition to put in the Notion text:
- The gradient is the slope of the loss. Walking against it lowers the loss.
- $\partial L/\partial b$ is the average error: if the line is too high on average, lower b.
- $\partial L/\partial m$ weights each error by x: errors at large x tell you to change the slope.
- Too large an $\eta$ overshoots and the loss grows; too small is slow.

## 3. A single ReLU neuron

Add a ReLU after the linear step:

$$z_i=w\,x_i+b,\qquad a_i=\mathrm{ReLU}(z_i)=\max(0,z_i),\qquad \hat y_i=a_i$$

The ReLU slope is 1 where z > 0 and 0 where z < 0:

$$\mathrm{ReLU}'(z)=\begin{cases}1 & z>0\\ 0 & z<0\end{cases}$$

Chain rule, one more link than before ($L\to\hat y\to a\to z\to w$):

$$\frac{\partial L}{\partial w}=\frac2N\sum_i e_i\cdot\mathrm{ReLU}'(z_i)\cdot x_i\qquad
\frac{\partial L}{\partial b}=\frac2N\sum_i e_i\cdot\mathrm{ReLU}'(z_i)$$

Same update rule as before. What is new: **points where z < 0 contribute no gradient** (ReLU is flat there), so only the points on the "on" side teach the neuron. A neuron that is "off" for every data point gets no gradient at all and never recovers (a "dead" neuron). That is exactly why `fit_v.py` starts each kink at a random data point.

Shape: ReLU(w x + b) is flat, then a straight line that starts at the kink $x=-b/w$. One ReLU can make one arm of the V.

## 4. A ReLU layer (what `fit_v.py` fits)

J ReLU units in a hidden layer, then a linear sum:

$$z_{ij}=w_j x_i+b_j,\quad a_{ij}=\max(0,z_{ij}),\quad \hat y_i=\sum_{j=1}^{J} v_j\,a_{ij}+c$$

(In `fit_v.py`: `w1`=w, `b1`=b, `w2`=v, `b2`=c. `nn.Linear(1, hidden)` is the first equation, `nn.ReLU()` the second, `nn.Linear(hidden, 1)` the third.)

With $e_i=\hat y_i-y_i$ and loss $L=\frac1N\sum e_i^2$:

$$\frac{\partial L}{\partial c}=\frac2N\sum_i e_i\qquad
\frac{\partial L}{\partial v_j}=\frac2N\sum_i e_i\,a_{ij}$$

$$\frac{\partial L}{\partial w_j}=\frac2N\sum_i e_i\,v_j\,\mathrm{ReLU}'(z_{ij})\,x_i\qquad
\frac{\partial L}{\partial b_j}=\frac2N\sum_i e_i\,v_j\,\mathrm{ReLU}'(z_{ij})$$

Read the last two as: error, times the output weight $v_j$ (how much unit j matters), times the ReLU switch (is it on?), times the input. That is backpropagation: the error flows backward through the output weight, through the ReLU, to the input weight.

Why two units make a V (use this in the write-up): with kink at $x_0$,

$$\mathrm{ReLU}(x-x_0)+\mathrm{ReLU}(x_0-x)=|x-x_0|$$

so $\hat y = 2\,\mathrm{ReLU}(x-60)+2\,\mathrm{ReLU}(60-x)$ is the V in your data. Extra units (`--hidden 4`) just give the fit more flexibility to match the noise and bend.

`fit_v.py` lines that map to the math:
- `loss = loss_fn(model(x), y)` is $L$ (forward pass).
- `loss.backward()` computes all the partial derivatives above automatically (autograd = the chain rule).
- `opt.step()` is $\text{param}\leftarrow\text{param}-\eta\,\partial L/\partial\text{param}$.
- `lr=0.05` is $\eta$. Inputs are scaled to mean 0, std 1 so one $\eta$ works for every weight.

## 5. The Excel spreadsheet: 5 iterations on m and b

Which model? The assignment says m & b, so the single linear neuron from section 2. A line cannot fit the whole V (section 1), so use a piece of the data a line *can* fit, and say so in the write-up. Good choices:
- **Right arm only** (x > 65, about 46 points, expected answer m about 2, b about -118), or
- a handful of points (5 to 8) from the right arm, which keeps the sheet small and every step visible.

**Important practical point:** with raw x up to 100, the slope gradient is huge and $\eta$ must be tiny. Easiest: divide x by 100 (and optionally y by 100) in a column first. Or keep raw numbers and use a small $\eta$ (try 0.0001 and watch the loss).

Suggested layout:

| Cells | Contents |
|---|---|
| Parameter table (top) | N, learning rate $\eta$, and rows for iteration 0 to 5 with columns: iteration, m, b, loss, dL/dm, dL/db |
| Row "iteration 0" | your starting guess for m and b (any numbers, e.g. m = 0, b = 0) |
| Data block (below) | columns: x, y, y-hat = m*x+b, error e = y-hat - y, e squared, e*x |
| Totals | loss = AVERAGE(e squared); dL/dm = 2*AVERAGE(e*x); dL/db = 2*AVERAGE(e) |
| Next row of params | m_new = m - eta * dL/dm; b_new = b - eta * dL/db |

Formulas (adjust cell names to your layout):
- `y-hat`: `=$m_cell * x_cell + $b_cell`
- `e`: `=yhat_cell - y_cell`
- `loss`: `=AVERAGE(esq_range)`
- `dL/dm`: `=2*AVERAGE(ex_range)`, `dL/db`: `=2*AVERAGE(e_range)`
- update: `=m_prev - eta * dLdm_prev`

To get 5 iterations without copying the data block 5 times: make the data block's columns once for each iteration (iteration 1 uses the m and b from iteration 0's update, and so on), or build the block once and copy the whole block five times to the right. Either is fine; showing all 5 side by side is clearer for the Notion screenshot.

What to show: the loss going down each iteration, and the line (m, b) moving toward the data. A small scatter plot with the 5 lines is a nice touch.

If you want, send me your starting values and learning rate and I will compute the 5 iterations in Python as an **answer key** to check your sheet against. I won't build the sheet for you.

## 6. What to say about the two plots from `fit_v.py`

- **Data and fit:** the orange curve follows the V: flat-sloped pieces joined at the bottom near x = 60. The RMS error (about 2) is comparable to the noise in the data, so the net has captured the shape and not the noise.
- **Loss vs step (log scale):** the loss drops quickly at first, when the weights are far from good and the gradients are large, then flattens as the fit gets close. It never reaches exactly zero because the data is noisy.
- Mention the random restarts: gradient descent can get stuck (for example both ReLUs facing the same way, which can only make one arm), so the script tries 5 starts and keeps the best.

## 7. Checklist for the Notion section

- [ ] Data description and the "a line cannot fit a V" contrast
- [ ] Single neuron: model, loss, gradients, update rule (section 2)
- [ ] ReLU neuron and ReLU layer: model and gradients (sections 3-4)
- [ ] Excel screenshot or embedded sheet with 5 iterations
- [ ] Plot 1: data and fit
- [ ] Plot 2: loss vs epoch
- [ ] A sentence connecting this to the finger lock: same ideas, more weights and a convolution layer
