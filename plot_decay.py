import matplotlib.pyplot as plt
import numpy as np

d = np.linspace(0, 3, 500)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: fixed p=3, varying scale s
ax = axes[0]
ax.set_title("p=3, varying scale s")
for s in [1, 2, 3, 5]:
    r = d / s
    sg = np.exp(-(r ** 3))
    rat = 1.0 / (1.0 + r ** 3)
    ax.plot(d, sg, "-", linewidth=2, label=f"SuperGaussian s={s}")
    ax.plot(d, rat, "--", linewidth=2, label=f"Rational s={s}")
ax.axhline(0.5, color="gray", ls=":", alpha=0.4)
ax.set_xlabel("d / λ  (distance in wavelengths)")
ax.set_ylabel("Decay factor")
ax.set_ylim(-0.05, 1.05)
ax.legend(fontsize=7, ncol=2)
ax.grid(True, alpha=0.3)

# Right: fixed s=1, varying p
ax = axes[1]
ax.set_title("s=1, varying power p")
for p in [2, 3, 4, 6]:
    sg = np.exp(-(d ** p))
    rat = 1.0 / (1.0 + d ** p)
    ax.plot(d, sg, "-", linewidth=2, label=f"SuperGaussian p={p}")
    ax.plot(d, rat, "--", linewidth=2, label=f"Rational p={p}")
ax.axhline(0.5, color="gray", ls=":", alpha=0.4)
ax.set_xlabel("d / λ  (distance in wavelengths)")
ax.set_ylabel("Decay factor")
ax.set_ylim(-0.05, 1.05)
ax.legend(fontsize=7, ncol=2)
ax.grid(True, alpha=0.3)

plt.suptitle("SuperGaussian (solid) vs Rational (dashed) decay", fontsize=13)
plt.tight_layout()
plt.savefig("decay_comparison.png", dpi=150)
plt.show()
