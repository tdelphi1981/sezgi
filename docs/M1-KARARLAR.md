# sezgi M1 — Yürütme Kararları ve Devir Notları

**Tarih:** 2026-08-28 · **Durum:** M1 main'e merge edildi (`b575b0c`), 17/17 görev + final inceleme temiz.
**Doğrulanmış:** 65/65 Rust testi, 8/8 pytest, `cargo build --workspace --locked` ✓, çapraz-dil altın değer `c05f7cc7f16d3d8b` iki dilde bit-uyumlu.
**Referanslar:** spec `docs/superpowers/specs/2026-08-27-sezgi-design.md` · plan `docs/superpowers/plans/2026-08-27-sezgi-m1.md`

## Yürütme sırasında verilen kararlar (rulings)

Plan/spec'in yanıtlamadığı yerlerde kontrolör olarak verdiğim kararlar. Yanlış bulduğun varsa geri alınabilir; her birinin "yanlışsa maliyeti" düşük tutuldu.

1. **DE donör indeksleri karşılıklı-farklı** — plan kodu yalnız `≠i` garantiliyordu; klasik DE tanımı esas alındı (`pick_distinct`, `crates/components/src/de.rs`). Altın değerler bu davranışla sabitlendi.
2. **T16'daki işlevsiz `py.allow_threads(|| ())` satırı atlandı** — dokümantasyon artığıydı.
3. **Bütçe < pop_size → `EmptyPopulation` hatası kabul edildi** — ad yanıltıcı ama davranış doğru; M2'de adlandırma cilası.
4. **`.gitignore` (/target) + Cargo.lock commit konvansiyonu eklendi** — planda yoktu; repo `--locked` derlenebilir tutuluyor (her bağımlılık değişikliğinde lock commit'e girer).
5. **"Cargo.lock'ta zmij uydurma bağımlılık" bulgusu kanıtla reddedildi** — `zmij` gerçek crate (serde_json'un ryu yerine geçen float-format bağımlılığı); crates.io + `--locked` derleme ile doğrulandı, final incelemeci de teyit etti.
6. **Boş `stages` listesi doğrulamada reddediliyor** (`SpecError::EmptyStages`) — plan kodu korumasızdı, motorda sonsuz döngüye yol açıyordu; spec'in "koşu öncesi anlaşılır hata" ilkesiyle çözüldü.
7. **`gen/de` için pop_size ≥ 4 net `assert!` ile korunuyor** — sessiz asılma yerine fail-fast; metadata-tabanlı (`min_pop`) koşu-öncesi kontrol M2 iyileştirmesi.
8. **`sezgi-components`'e `sezgi-problems` dev-bağımlılığı onaylandı** — altın test `BbobProblem` gerektiriyordu; oluşan dev-dep döngüsü (problems↔components) Cargo'da meşru, `cargo metadata --locked` doğrulandı.
9. **`RunResult.best_f` global-en-iyi olarak düzeltildi** (final inceleme bulgusu) — elitist-olmayan replacer'larda (PSO) popülasyon-en-iyisi yanlış raporlanıyordu; takip RNG tüketmediği için altın değerler korundu (`engine.rs`, test: `non_elitist_replacer_keeps_global_best`).
10. **Best1'deki kullanılmayan `r1` RNG çekimi bilerek düzeltilmedi** — kaldırmak RNG tüketim sırasını değiştirip altın yörüngeyi bozar; Best1 kendi altın testini alırken (M2) ele alınacak.
11. **numpy sapması kabul edildi** — plan "numpy crate + zero-copy" diyordu, M1 `list[list[float]]` ile gitti; zero-copy NumPy planın kendi M2 performans maddesi.
12. **Evaluator batch-uzunluk kontrolü panik olarak eklendi** — `from_callable` ile Python'dan ulaşılabilir hale gelmişti; yanlış uzunlukta net mesajlı panik (PyO3 bunu Python istisnasına çevirir), temiz PyErr köprüsü M2.
13. **Callable problem + `log_dir` → ValueError** — sessiz parametre yutma yerine açık hata ("log_dir yalnız yerleşik (bbob) problemlerde destekleniyor").

## M2'ye devreden iş listesi (ertelenen bulgular)

**M2'de zorunlu:**
- [ ] T14: sıfır-değerlendirmeli koşuda meta JSON `best.y=null` üretiyor — IOHanalyzer/IOHinspector entegrasyonundan ÖNCE guard + test.
- [ ] BBOB tamamlama: f5–f24 + `cocoex` çapraz doğrulama (plan bunu ayrı "BBOB tamamlama" planına ertelemişti).
- [ ] NumPy zero-copy toplu değerlendirme + callback istisnaları için temiz PyErr köprüsü + Bbob dalında `allow_threads` (GIL koşu boyunca tutuluyor).

**M2'de ele alınacak cilalar:**
- [ ] Dağılım parametre doğrulaması (Levy alpha, StudentT nu — şu an sessiz NaN riski) + Levy/StudentT/Laplace istatistiksel testleri.
- [ ] `EmptyPopulation` adlandırması; `ManifestError::Engine`'in SpecError/EngineError'ı string'e indirgemesi.
- [ ] RNG yol çakışması: boundary `[run_id,1000]` ile 500. stage replacer'ı çakışır (engine.rs'de yorumla işaretli) — motor yolları ayrılırken ofseti büyüt.
- [ ] Best1: fark vektörü indekslerinin best'i dışlamaması + kullanılmayan r1 çekimi (madde 10) + Best1 yakınsama/altın testi.
- [ ] PsoCommit çift-geçiş/klon optimizasyonu; sınır kırpması sonrası hız sıfırlama seçeneği.
- [ ] Küçükler: SearchSpace dejenere blok (k=0, n=0) reddi; negatif tournament_k'nin sessizce varsayılana düşmesi; `next_below`/`Clamp`/`gen-step` birim testleri; CallableProblem'da Float-dışı blok için `unreachable!`.

## Örnek üçlemesi (kullanıcı onaylı, M2/M3 kapsamına girecek)

`examples/` altında algoritma başına üç eş implementasyon. Hedef katalog GENİŞ (~15 algoritma; kullanıcı özellikle geniş olmasını istedi) — her spec varyantı aynı zamanda M3'ün etiketli ön ayar kataloğuna girer ve yanlılık-taraması vitrininde koşulur:

**Katalog (aile + bilinen eşdeğerlik/yanlılık referansıyla):**
- PSO-ailesi şemaları: GWO (Grey Wolf), WOA (Whale), GOA (Grasshopper), MFO (Moth-Flame), SSA (Salp Swarm), ALO (Antlion), SCA (Sine-Cosine), FA (Firefly), BA (Bat) — Camacho-Villalón 2023 eşdeğerlik analizleri; birçoğu Kudela merkez-yanlılık listesinde.
- Lévy-uçuşu vitrini (dağılım-parametreli bileşenler): CS (Cuckoo Search), FPA (Flower Pollination), HHO (Harris Hawks — Rajwar-Deep imza testinde güçlü yanlılık örneği).
- ES/GA-eşdeğerleri: HS (Harmony Search ≡ ES özel hali, Weyland 2015), TLBO (Teaching-Learning), JAYA (parametresiz), ABC (Artificial Bee Colony), GSA (Gravitational Search).

Öncelik sırası: GWO, WOA, HS, CS, GOA ilk dalga (M2 sonu Python); kalanlar M3'te ön ayarlarla birlikte. Her algoritma için:
- **`<algo>_pure.py`** — saf Python ask/tell (aday üretimi NumPy'da; çekirdek yalnız değerlendirme+günlükleme+sayaç): "mevcut kodu taşımak kolay" kapısı.
- **`<algo>_pure.R`** — aynısı saf R ask/tell.
- **`<algo>_spec.py` / `<algo>_spec.R`** — ÖNERİLEN: bileşen spec'i (~15 satır konfigürasyon, tam Rust hızı, eşdeğerlik referansı yorumda).

Getiriler: (1) üç varyantı deney koşturucuda koşturup "saf Python vs spec: ~Nx" hız tablosu → makale performans bölümü; (2) saf ve spec varyantların istatistiksel eşdeğerlik testi → spec çevirisinin doğrulaması; (3) "kendi algoritmanı yaz" dokümantasyonunun hazır basamağı.

Dürüstlük notu (örneklerin başına yazılacak): ask/tell varyantı kendi RNG'sini (numpy/R) kullandığından spec varyantıyla bit-uyumlu DEĞİL, istatistiksel eşdeğer; bit-uyum garantisi yalnız çekirdek-içi yürütmede.

Zamanlama: Python örnekleri M2 sonu (deney koşturucu gerektirir), R örnekleri R ön yüzüyle, üçlü karşılaştırma M3'te metafor ön ayarlarıyla tamamlanır.

## M2a tamamlandı (2026-08-28, main'e merge: d057700)

**Teslim:** sezgi-bbob 24/24 fonksiyon (T_osz/T_asy/Λ/f_pen dönüşümleri, fid-politikalı x_opt, Q rotasyonu, Gallagher verileri, 24×3 süit özellik testleri, M1 drift bit-pinleri) + IOH best.y=null koruması (IohFinish) + dağılım parametre doğrulaması + numpy callback + PyErr köprüsü + allow_threads + BudgetSmallerThanPopulation ayrımı. 12 commit, final tüm-dal incelemesi temiz, Rust hatasız + pytest 9/9, altınlar bit-uyumlu.

**M2a ruling'leri:**
1. needs_q/needs_r tabloları plandan düzeltildi (Q: {6,7,13,15,16,17,18,23,24}; R: fid ∉ {1,5,20}) — dispatch'lerin kullanmadığı çekimler kaldırıldı.
2. "cocoex çapraz doğrulama" → analitik altın/özellik testleri (M1'in COCO-birebir-değil ruling'iyle tutarlı); sapmalar kodda "sezgi-bbob sadeleştirmesi" etiketiyle belgelendi (f12 çift-R, f20 varyant formül, f21/22 rastgele alpha + köşegen C).
3. Panik-hook stderr gürültüsü bilinçli kabul (global hook takası süreç-geneli/yarışlı — kodda NOT yorumu).
4. "Zero-copy" girişte kopyalı (from_vec2), dönüşte zero-copy — kod yorumu dürüst, aksiyon yok.
5. needs_r(4) düzeltmesi (f4 R çekimi kullanılmıyor) M2b'ye ertelendi — instance-akışı kararı yayın öncesi 24-fid bit-pin tablosuyla birlikte tek seferde.

**M2b backlog'una eklenenler:** 24-fid bit-pin tablosu; needs_r(4) kararı; skipped_empty_runs'ın Python'a yüzeylenmesi; eski dispatch kollarının shift() helper'ına geçirilmesi; Levy/StudentT/Laplace istatistiksel testleri.

## Okulda devam ederken

- Depo temiz, her şey `main`'de. Python ortamı: `cd py-sezgi && python -m venv .venv && .venv/bin/pip install maturin pytest && .venv/bin/maturin develop --release && .venv/bin/python -m pytest tests/ -v` (venv commit'li değil, okulda yeniden kurulur).
- Doğal sıradaki adım: **M2 planı** (R ön yüzü [savvy] + istatistik modülü + deney koşturucu + CMA-ES/SHADE/L-SHADE/SA/Nelder-Mead + yukarıdaki zorunlular) — `superpowers:writing-plans` ile, spec Bölüm 11 yol haritasından.
- Alternatif önce: GitHub'a push + CI (`cargo build --workspace --locked` kontrolü dahil — final incelemenin önerisi).
- Yürütme tercihi: subagent-driven (kalıcı bellekte kayıtlı).
