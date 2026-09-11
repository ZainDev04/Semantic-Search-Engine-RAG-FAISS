from fetch_data import clean_body, clean_title


def test_clean_title_strips_pr_suffix():
    assert clean_title("Fix streaming bug (#1234)") == "Fix streaming bug"
    assert clean_title("Fix streaming bug (#1234)   ") == "Fix streaming bug"


def test_clean_title_keeps_parenthetical_that_is_not_a_pr_number():
    assert clean_title("Support (nested) parquet") == "Support (nested) parquet"
    assert clean_title("Fix #1234 in loader") == "Fix #1234 in loader"


def test_clean_body_drops_git_trailers_only():
    body = (
        "Rebatch the arrow source before formatting.\n"
        "\n"
        "Co-authored-by: Someone <a@b.c>\n"
        "Signed-off-by: Someone Else <d@e.f>\n"
        "  Reviewed-by: R <r@r.r>\n"
    )
    assert clean_body(body) == "Rebatch the arrow source before formatting."


def test_clean_body_made_only_of_trailers_becomes_empty():
    assert clean_body("Co-authored-by: A <a@a.a>\nSigned-off-by: B <b@b.b>") == ""


def test_clean_body_is_case_insensitive_for_trailers():
    assert clean_body("real text\nCO-AUTHORED-BY: X <x@x.x>") == "real text"


def test_clean_title_strips_stacked_pr_suffixes():
    # A cherry-pick onto a release branch appends a second PR number.
    assert clean_title("Fix loader (#8241) (#8300)") == "Fix loader"
    assert clean_title("Fix loader (#1) (#2) (#3)") == "Fix loader"
