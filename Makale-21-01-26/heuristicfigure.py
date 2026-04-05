import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Data ──────────────────────────────────────────────────────────────────────
suppliers = [100, 150, 200, 250, 300]

inventory  = [13732.10, 18851.35, 24187.26, 30595.70, 35718.26]
assignment = [4465.36,  8898.40,  17961.82, 19953.34, 26328.61]
waste      = [9286.70,  6834.13,  1488.08,  1791.20,  2177.00]

# ── Style ─────────────────────────────────────────────────────────────────────
BLUE   = '#185FA5'
GREEN  = '#0F6E56'
CORAL  = '#993C1D'
BLACK  = '#000000' # Eksenler için tam siyah

plt.rcParams.update({
    'font.family':       'serif',
    'font.size':         9,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.linewidth':    0.8,   # Eksen çizgisi kalınlığı
    'axes.edgecolor':    BLACK, # Eksen çizgisi rengi (Siyah)
    'axes.grid':         False, # Izgara kapalı
    'xtick.color':       BLACK, # X ekseni sayıları siyah
    'ytick.color':       BLACK, # Y ekseni sayıları siyah
    'axes.labelcolor':   BLACK, # "Number of suppliers" gibi etiketler siyah
})

marker_kw = dict(marker='o', markersize=6, linewidth=2.0, markeredgewidth=0)

# ── Figure ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(12, 4.0))
fig.subplots_adjust(wspace=0.38)

panels = [
    (axes[0], inventory,  BLUE,  '(a) Inventory holding cost', 'Inventory holding cost (€)'),
    (axes[1], assignment, GREEN, '(b) Assignment cost',         'Assignment cost (€)'),
    (axes[2], waste,      CORAL, '(c) Waste cost',              'Waste cost (€)'),
]

for ax, data, color, title, ylabel in panels:
    # Çizgi ve noktalar kendi renginde kalıyor
    ax.plot(suppliers, data, color=color, **marker_kw)
    
    # Eksen etiketleri
    ax.set_xlabel('Number of suppliers')
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9, loc='left', pad=6, color='#3d3d3a')
    ax.set_xticks(suppliers)
    
    # Y ekseni formatı
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f'€{x/1000:.0f}k')
    )
    
    # Nokta üzerindeki değer etiketleri (Grafik rengiyle uyumlu)
    for x, y in zip(suppliers, data):
        ax.annotate(f'€{y/1000:.1f}k', xy=(x, y),
                    xytext=(0, 8), textcoords='offset points',
                    ha='center', fontsize=7.5, color=color)

# ── Save ───────────────────────────────────────────────────────────────────────
plt.savefig(r"C:\Users\gizem\OneDrive\Belgeler\GitHub\gizem_tez\Makale-21-01-26\cost_components.pdf",
            dpi=300, bbox_inches='tight', format='pdf')
plt.savefig(r"C:\Users\gizem\OneDrive\Belgeler\GitHub\gizem_tez\Makale-21-01-26\cost_components.png",
            dpi=300, bbox_inches='tight', format='png')
print("Saved.")