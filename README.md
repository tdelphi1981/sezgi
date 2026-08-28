# sezgi

Rust çekirdekli, bileşen-tabanlı, Python ve R ön yüzlü meta-sezgisel
optimizasyon kütüphanesi. Tasarım: `docs/superpowers/specs/2026-08-27-sezgi-design.md`.

## Hızlı başlangıç (Python)

    import sezgi

    problem = sezgi.bbob(fid=1, dim=10, instance=1)
    spec = sezgi.presets.de_rand_1(pop_size=50, budget=20_000)
    result = sezgi.solve(spec, problem, master_seed=42, log_dir="logs/")
    print(result["best_f"])   # IOH-format günlük logs/ altında

Kendi probleminiz (toplu değerlendirme — popülasyon başına tek çağrı):

    import numpy as np

    def f(X):                       # X: np.ndarray (n, d) float64
        return ((X - 1.0) ** 2).sum(axis=1)

    problem = sezgi.from_callable(f, lo=-5.0, hi=5.0, dim=10)

## Geliştirme

    cargo test --workspace --release        # Rust testleri
    cd py-sezgi && maturin develop && pytest # Python testleri

Durum: M1 (çekirdek + Python). M2a (sezgi-bbob 24/24 BBOB fonksiyonu). M2: R ön yüzü + istatistik; M3: yanlılık taraması + çok-amaçlı. Lisans: MIT.
