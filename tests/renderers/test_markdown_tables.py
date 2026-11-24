"""Table rendering regressions; run without models via python -m unittest."""

import unittest

from marker.renderers.markdown import MarkdownRenderer


class MarkdownTableTests(unittest.TestCase):
    def render(self, html, **config):
        return MarkdownRenderer(config).md_cls.convert(html).strip()

    def table_rows(self, markdown):
        # Ignore alignment padding while checking the actual GFM row structure.
        return [
            [cell.strip() for cell in line.strip().strip("|").split("|")]
            for line in markdown.splitlines()
            if line.startswith("|")
        ]

    def test_single_row_continuation_keeps_three_data_cells(self):
        markdown = self.render("""
            <table><tr>
            <td><ul><li>Signatures on System generated documents</li></ul></td>
            <td><ul><li>Policy Schedule</li><li>Endorsements</li>
            <li>Final Invoice</li><li>Credit /Debit Notes</li>
            <li>Claim Experience</li></ul></td>
            <td><ul><li>MUM S.E.D or Mobility Sector CEO</li></ul></td>
            </tr></table>
        """)
        for tag in ("<ul>", "<li>", "</li>", "</ul>"):
            self.assertNotIn(tag, markdown)
        self.assertEqual(len(markdown.splitlines()), 3)
        rows = self.table_rows(markdown)
        self.assertEqual(rows[0], ["", "", ""])
        self.assertTrue(all(set(cell) == {"-"} for cell in rows[1]))
        self.assertEqual(
            rows[2],
            [
                "• Signatures on System generated documents",
                "• Policy Schedule<br>• Endorsements<br>• Final Invoice<br>• Credit /Debit Notes<br>• Claim Experience",
                "• MUM S.E.D or Mobility Sector CEO",
            ],
        )

    def test_page_boundaries_and_following_heading_survive(self):
        markdown = self.render(
            """
            <div class="page" data-page-id="53">
            <table><tr><th>Authority</th><th>Document</th><th>Signature</th></tr>
            <tr><td>Authority level</td><td>Bail Bond</td><td>CEO</td></tr></table>
            </div><div class="page" data-page-id="54">
            <table><tr><td>Signatures</td><td>Policy Schedule</td><td>MUM S.E.D</td>
            </tr></table></div><div class="page" data-page-id="55">
            <h1>SCHEDULE (H): COMPUTATION OF PREMIUM RATES</h1></div>
        """,
            paginate_output=True,
        )
        self.assertLess(markdown.index("{53}"), markdown.index("Bail Bond"))
        self.assertLess(markdown.index("Bail Bond"), markdown.index("{54}"))
        self.assertLess(markdown.index("{54}"), markdown.index("Signatures"))
        self.assertLess(markdown.index("Signatures"), markdown.index("{55}"))
        self.assertIn("# SCHEDULE (H): COMPUTATION OF PREMIUM RATES", markdown)
        continuation = markdown.split("{54}")[1].split("{55}")[0]
        self.assertEqual(
            self.table_rows(continuation)[2],
            ["Signatures", "Policy Schedule", "MUM S.E.D"],
        )

    def test_explicit_header_has_one_separator(self):
        markdown = self.render(
            "<table><tr><th>Authority</th><th>Limit</th></tr></table>"
        )
        self.assertEqual(len(markdown.splitlines()), 2)
        self.assertEqual(self.table_rows(markdown)[0], ["Authority", "Limit"])

    def test_multiple_data_rows_are_not_promoted_to_headers(self):
        rows = self.table_rows(
            self.render(
                "<table><tr><td>A</td><td>B</td></tr><tr><td>C</td><td>D</td></tr></table>"
            )
        )
        self.assertEqual(rows[0], ["", ""])
        self.assertEqual(rows[2:], [["A", "B"], ["C", "D"]])

    def test_paragraphs_and_ordered_lists_stay_inside_cell(self):
        markdown = self.render("""
            <table><tr><td><p>Approval</p><ol start="3"><li><p>Review</p></li>
            <li value="7">Sign</li></ol><p>Done</p></td></tr></table>
        """)
        self.assertIn("Approval<br>3. Review<br>7. Sign<br>Done", markdown)
        self.assertEqual(len(self.table_rows(markdown)[2]), 1)

    def test_bullet_is_not_duplicated_and_nested_lists_keep_text(self):
        markdown = self.render("""
            <table><tr><td><ul><li>• Parent<ul><li>Child</li></ul></li>
            <li>Second</li></ul></td></tr></table>
        """)
        self.assertIn("• Parent<br>• Child<br>• Second", markdown)
        self.assertNotIn("• •", markdown)

    def test_inline_formatting_and_math_are_preserved(self):
        markdown = self.render(
            "<table><tr><td>54<i>.45</i>67<br>89<math>x</math></td></tr></table>"
        )
        self.assertIn("54 <i>.45</i> 67<br>89 $x$", markdown)

    def test_rowspan_reserves_columns_on_following_row(self):
        markdown = self.render("""
            <table><tr><td rowspan="2">Authority</td></tr>
            <tr><td>Limit</td></tr></table>
        """)
        self.assertEqual(
            self.table_rows(markdown)[2:], [["Authority", ""], ["", "Limit"]]
        )

    def test_colspan_keeps_column_alignment(self):
        markdown = self.render("""
            <table><tr><th colspan="2">Authority</th><th>Limit</th></tr>
            <tr><td>A</td><td>B</td><td>C</td></tr></table>
        """)
        rows = self.table_rows(markdown)
        self.assertEqual(rows[0], ["Authority", "", "Limit"])
        self.assertEqual(rows[2], ["A", "B", "C"])

    def test_html_table_option_preserves_original_structure(self):
        html = "<table><tr><td><ul><li>Policy</li></ul></td></tr></table>"
        self.assertEqual(self.render(html, html_tables_in_markdown=True), html)

    def test_empty_table_has_no_separator_rows(self):
        self.assertEqual(self.render("<table></table>"), "")


if __name__ == "__main__":
    unittest.main()
