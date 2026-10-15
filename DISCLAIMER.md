# ⚠️ Disclaimer: read this before you use critflaw for anything

**critflaw is research and development software. It is not verified, not validated, not certified, not
complete and not fit for any real-world purpose.**

## Do not use it for real decisions

Do **not** use critflaw, any of its results, or anything derived from them:

- to accept, reject, repair, re-rate, inspect or continue operating any real structure, component or plant;
- in design, fitness-for-service, engineering critical assessment, integrity management or life extension;
- in any safety case, safety-related or safety-critical application, or anything on which a person's health,
  life, property or the environment could depend;
- in any regulatory, legal, contractual, insurance or certification submission;
- as a substitute for the published assessment procedures, their licensed texts, qualified software, or the
  judgement of a competent, qualified engineer.

## What "not verified or validated" means here

- **No independent verification and validation (V&V) has been carried out.** The library's own test suite,
  partner-derived oracles and internal reviews are development checks, not V&V. Passing them proves nothing about
  correctness for a real assessment.
- **Equations and coefficients may be wrong.** They were transcribed by hand from a licensed standard and its
  OCR conversion; transcription errors have been found and fixed before, and others may remain. Most items have
  **not** been checked against their primary references (see `docs/unsourced_items.md`).
- **Many results rest on equations whose status is only `self-consistent`**, which cannot detect an error in the
  source itself. The trace of every result lists this; `Assessment.run(strict=True)` refuses such results, and
  today it refuses most of them.
- **The finite-element backend and the Option 3 emulator are experimental.** Family-level FE verification is
  incomplete, the emulator missed its accuracy target, and the end-to-end probabilistic test with the emulator
  (V10) **did not pass**.
- **Probabilities are model outputs, not facts.** They depend entirely on the input distributions you choose,
  exclude model uncertainty unless you add it, and the rare-event estimates are explicitly conditional.
- **The API, results and behaviour will change without notice.** This is pre-alpha software in active
  development.

## Standards and intellectual property

critflaw implements methods of the failure assessment diagram type described in published procedures. It is
**not** an implementation of, endorsed by, or affiliated with BSI, BS 7910, R6, API 579/ASME FFS-1 or any
standards body. Some coefficient sets and tables in the source were transcribed from a licensed copy of a
standard; their redistribution is **not cleared** (the library's own `critflaw.kb.core.release_audit` blocks
them). If you hold rights in that material and object to its presence, open an issue and it will be removed.

## No warranty, no liability

The software is provided "as is", without warranty of any kind, express or implied, including fitness for a
particular purpose, accuracy or non-infringement, under the terms of the BSD-3-Clause licence (`LICENSE`). The
authors and contributors accept **no liability** for any loss, damage, injury or consequence of any kind arising
from its use or misuse.

**If you are making a real engineering decision, close this repository and use qualified tools, the licensed
standard and a competent engineer.**
