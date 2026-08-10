"""Target phrase extraction from natural-language mission instructions."""

from airsim_benchmark.core.target_phrase import extract_target


def test_fly_toward_red_car():
    info = extract_target("Fly toward the red car parked in the neighborhood")
    assert info.phrase == "red car"
    assert info.wants_land is False


def test_land_on_rooftop_near_red_car():
    info = extract_target("Land on the closest rooftop to the red car")
    assert info.phrase == "rooftop"
    assert info.wants_land is True


def test_navigate_to_two_story_house():
    info = extract_target("Navigate to the two-story house")
    assert info.phrase == "two-story house"
    assert info.wants_land is False


def test_descend_and_land_on_rooftop():
    info = extract_target("Descend and land on the rooftop near the red car")
    assert info.phrase == "rooftop"
    assert info.wants_land is True


def test_fly_toward_white_house():
    info = extract_target("Fly toward the white house with the big yard")
    assert info.phrase == "white house"
    assert info.wants_land is False


def test_navigate_to_intersection():
    info = extract_target("Navigate to the intersection ahead")
    assert info.phrase == "intersection"
    assert info.wants_land is False


def test_fly_to_mailbox():
    info = extract_target("Fly to the mailbox at the end of the driveway")
    assert info.phrase == "mailbox"
    assert info.wants_land is False


def test_unknown_landmark_fallback():
    info = extract_target("Go to the mysterious blue tower")
    assert "blue tower" in info.phrase


def test_land_on_ground():
    info = extract_target("Land safely on the ground")
    assert info.wants_land is True
    assert info.wants_ground_land is True


def test_land_on_rooftop_ground_false():
    info = extract_target("Land on the rooftop")
    assert info.wants_land is True
    assert info.wants_ground_land is False


def test_multi_query_includes_broader_terms():
    info = extract_target("Land on the rooftop")
    assert "rooftop" in info.multi_query
    assert "house top view" in info.multi_query


def test_multi_query_red_car_includes_vehicle():
    info = extract_target("Fly toward the red car")
    assert "red car" in info.multi_query
    assert "vehicle" in info.multi_query


def test_reference_phrase_extracted():
    info = extract_target("Land on the closest rooftop to the red car")
    assert info.reference_phrase == "red car"


def test_no_reference_when_absent():
    info = extract_target("Fly toward the red car")
    assert info.reference_phrase is None


def test_multi_query_unknown_phrase():
    info = extract_target("Fly to the blue tower")
    assert "blue tower" in info.multi_query
