from unittest import TestCase
from src.nodes.general_context_node import general_context_node


class general_context_node_tests(TestCase):

    def test_original_question_none(self):

        with self.assertRaises(Exception) as exception:
            general_context_node(None, "")

        self.assertEqual(
            str(exception.exception),
            "Invalid question.",
            "Original Question cannot be None"
        )

    def test_original_question_empty(self):

        with self.assertRaises(Exception) as exception:
            general_context_node("", "")

        self.assertEqual(
            str(exception.exception),
            "Invalid question.",
            "Original Question cannot be empty string"
        )

    def test_user_response_none(self):

        with self.assertRaises(Exception) as exception:
            general_context_node("a", None)

        self.assertEqual(
            str(exception.exception),
            "Invalid user response.",
            "User response cannot be None"
        )

    def test_user_response_empty(self):

        with self.assertRaises(Exception) as exception:
            general_context_node("a", "")

        self.assertEqual(
            str(exception.exception),
            "Invalid user response.",
            "User response cannot be empty string"
        )

    def test_user_response_ask_for_answer_directly(self):

        res = general_context_node(
            "What is the best way to sort an array of integers?",
            "Give me the answer?"
        )

        self.assertEqual(
            res,
            False,
            "Asking for answer directly is out of context"
        )

    def test_user_response_out_of_context(self):

        res = general_context_node(
            "What is the best way to sort an array of integers?",
            "How are you?"
        )

        self.assertEqual(
            res,
            False,
            "Asking unrelated question should fail"
        )

    def test_user_response_with_another_question(self):

        res = general_context_node(
            "What is the best way to sort an array of integers?",
            "What time complexity should I achieve ?"
        )

        self.assertEqual(
            res,
            True,
            "Asking a related follow up question should pass."
        )

    def test_user_response_ask_for_example(self):

        res = general_context_node(
            "What is the best way to sort an array of integers?",
            "Can you give me an example ?"
        )

        self.assertEqual(
            res,
            True,
            "Asking for example should pass."
        )

    def test_user_response_being_rude(self):

        res = general_context_node(
            "What is the best way to sort an array of integers?",
            "Fuck you"
        )

        self.assertEqual(
            res,
            False,
            "Being rude should fail."
        )
