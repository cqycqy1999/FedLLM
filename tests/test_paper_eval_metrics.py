import unittest

from fedpost.evaluation.metrics import (
    evaluate_humaneval_completion,
    execute_finqa_program,
    extract_choice,
    numeric_equal,
    score_finqa_prediction,
    score_medqa_prediction,
)


class PaperEvalMetricsTest(unittest.TestCase):
    def test_medqa_choice_extraction(self):
        self.assertEqual(extract_choice("Final answer: C"), "C")
        self.assertEqual(extract_choice("probably c"), "C")
        self.assertTrue(score_medqa_prediction(
            "The answer is B.",
            {"answer_idx": "B", "options": [{"key": "A", "value": "x"}, {"key": "B", "value": "y"}]},
        ))

    def test_finqa_program_execution(self):
        self.assertEqual(execute_finqa_program("subtract(10, 4)"), 6.0)
        self.assertEqual(execute_finqa_program("subtract(10, 4), divide(#0, 3)"), 2.0)
        self.assertTrue(score_finqa_prediction("Final answer: 2.00001", "2"))
        self.assertTrue(numeric_equal("50%", "0.5"))

    def test_humaneval_pass_at_one_execution(self):
        prompt = "def add_one(x):\n"
        completion = "    return x + 1\n"
        test = "def check(candidate):\n    assert candidate(1) == 2\n"
        passed, detail = evaluate_humaneval_completion(prompt, completion, test, "add_one", timeout=1.0)
        self.assertTrue(passed, detail)


if __name__ == "__main__":
    unittest.main()
