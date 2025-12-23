import os
from dataclasses import asdict
from pprint import pprint

from core.reasoning.clinicalization.clinicalizer import Clinicalizer


# --------------------------------------------------
# CONFIG
# --------------------------------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if GEMINI_API_KEY is None:
    raise RuntimeError(
        "GEMINI_API_KEY not set. "
        "Export it before running this test."
    )

TEST_QUERIES = [
    # Pediatric infectious
    "11-year-old girl with high fever for 3 days, swollen tonsils with white patches and severe fatigue",

    # Cardiorespiratory
    "65-year-old male with chest pain, shortness of breath, cold sweats, and dizziness",

    # Dermatology
    "Young adult with bluish discoloration of fingertips and lips worsening in cold weather",

    # Negative / edge case
    "Patient complains of headache but no fever, no nausea, and no visual disturbances"
]


# --------------------------------------------------
# TEST RUNNER
# --------------------------------------------------
def run_test():
    print("\n🚀 Initializing Clinicalizer...\n")

    clinicalizer = Clinicalizer(
        gemini_api_key=GEMINI_API_KEY
    )

    for i, query in enumerate(TEST_QUERIES, start=1):
        print("=" * 90)
        print(f"TEST CASE {i}")
        print("=" * 90)
        print(f"\n📝 Raw Query:\n{query}\n")

        clinical_query = clinicalizer.clinicalize(query)

        # -------------------------
        # ENTITY EXTRACTION
        # -------------------------
        print("🔹 ENTITY EXTRACTION")
        pprint(asdict(clinical_query.entities))

        # Sanity checks
        assert clinical_query.entities.symptoms is not None
        assert clinical_query.entities.findings is not None
        assert clinical_query.entities.umls_cuis is not None

        # -------------------------
        # LLM ABSTRACTION
        # -------------------------
        print("\n🔹 CLINICAL ABSTRACTION (LLM)")
        pprint(asdict(clinical_query.abstraction))

        # Schema integrity checks
        abs_ = clinical_query.abstraction
        assert isinstance(abs_.chief_complaints, list)
        assert isinstance(abs_.key_symptoms, list)
        assert isinstance(abs_.clinical_syndromes, list)
        assert isinstance(abs_.risk_factors, list)
        assert isinstance(abs_.red_flags, list)
        assert isinstance(abs_.normalized_terms, dict)

        print("\n✅ Test case PASSED")

    print("\n🎉 ALL CLINICALIZATION TESTS PASSED SUCCESSFULLY\n")


# --------------------------------------------------
# ENTRY POINT
# --------------------------------------------------
if __name__ == "__main__":
    run_test()
