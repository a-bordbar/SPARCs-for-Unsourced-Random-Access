import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import norm
# In order to evaluate Q(x), use norm.sf(x) 
# In order to evaluate Q^{-1}(x), use norm.isf(x)
from scipy.stats import binom
from pathlib import Path



# This script reproduces fig.2 in the paper 
snr_vec_dB = np.array([10, 13, 15]) # SNR values of the effective channel (eta * Phat)in dB
snr_vec_lin = 10**(snr_vec_dB/10) # SNR values in linear scale
Ka = 300 # Number of active users
J = 12 # number of bits in each section

pfa_vec = np.logspace(-6, -1, 500)
p = 2.0 ** (-J)

#Now I plot the curves for each SNR value 
pmd_mat = np.zeros((len(snr_vec_lin), len(pfa_vec))) # Initialize the matrix to store PMD values
for i, snr_ in enumerate(snr_vec_lin):
    pmd_mat[i,:] = norm.sf(np.sqrt(snr_) - norm.isf(pfa_vec)) # This is Eq. (52) in the paper.
    
# --- Now I plot the curve for the exact channel without the OR-estimation --- #

# First, I need to define the Neymann-Pearson threshold for the LLR test. 

gamma_vec = np.logspace(-6, -1, 500)

#Now, for each value of gamma, I need to compute pfa and pmd. 




pmd_mat_exact = np.zeros((len(gamma_vec), len(pfa_vec))) # Initialize the matrix to store exact PMD values

#Next, I compute the weights

wk_numerator = binom.pmf(np.arange(1, Ka+1), n=Ka, p=p) # This is the numerator of the weights

wk_denominator =  binom.sf(0, n=Ka, p=p)
wk = wk_numerator / wk_denominator # This is the weight for each k value
kvec= np.arange(1, Ka+1) # This is the vector of k values
gamma_vec = norm.isf(pfa_vec)

for i, snr_ in enumerate(snr_vec_lin):

    # Matrix dimensions:
    #
    # kvec[:, None]  : (Ka, 1)
    # gamma_vec[None,:]: (1, number of P_FA values)
    #
    # Result: (Ka, number of P_FA values)
    pmd_given_k = norm.sf(
        kvec[:, None] * np.sqrt(snr_)
        - gamma_vec[None, :]
    )

    # Weighted average over k
    pmd_mat_exact[i, :] = np.sum(
        wk[:, None] * pmd_given_k,
        axis=0
    )
        
        
# ==========================================================
# Plot
# ==========================================================

plt.figure(figsize=(7, 5))

for i, snr_db in enumerate(snr_vec_dB):

    # Exact curve
    exact_line, = plt.loglog(
        pfa_vec,
        pmd_mat_exact[i, :],
        linewidth=1.8,
        label=f"Exact, {snr_db:.0f} dB"
    )

    # OR approximation: use the same color
    plt.loglog(
        pfa_vec,
        pmd_mat[i, :],
        linestyle="--",
        linewidth=1.8,
        color=exact_line.get_color(),
        label=f"OR approximation, {snr_db:.0f} dB"
    )

plt.xlabel(r"$P_{\mathrm{FA}}$")
plt.ylabel(r"$P_{\mathrm{MD}}$")

plt.xlim(1e-4, 1e-1)
plt.ylim(1e-6, 1)

plt.grid(
    True,
    which="major",
    alpha=0.25,
    linewidth=0.7
)

plt.legend()
plt.tight_layout()
plt.show()


# ==========================================================
# Save results to CSV
# ==========================================================

output_directory = Path("./data/fig2_data")
output_directory.mkdir(parents=True, exist_ok=True)

for i, snr_db in enumerate(snr_vec_dB):

    data = np.column_stack((
        pfa_vec,
        pmd_mat_exact[i, :],
        pmd_mat[i, :]
    ))

    filename = output_directory / f"fig2_{snr_db:.0f}dB.csv"

    np.savetxt(
        filename,
        data,
        delimiter=",",
        header="P_FA,P_MD_exact,P_MD_OR",
        comments="",
        fmt="%.18e"
    )

    print(f"Saved: {filename}")