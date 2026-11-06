from lmspell.trainers.utils.early_stopping import EarlyStopping


def test_tracks_the_best_epoch():
    stopper = EarlyStopping(epochs=5, patience=2, exp_name="exp")

    assert stopper.step(1.0, 0.9, "epoch_1") is False
    assert stopper.step(0.8, 0.7, "epoch_2") is False

    assert stopper.best_epoch == 2
    assert stopper.best_model == "epoch_2"


def test_stops_after_patience_epochs_without_improvement():
    stopper = EarlyStopping(epochs=10, patience=2, exp_name="exp")

    assert stopper.step(1.0, 0.5, "epoch_1") is False
    assert stopper.step(1.0, 0.6, "epoch_2") is False
    assert stopper.step(1.0, 0.7, "epoch_3") is True


def test_stops_at_the_last_epoch():
    stopper = EarlyStopping(epochs=2, patience=5, exp_name="exp")
    assert stopper.step(1.0, 0.9, "epoch_1") is False
    assert stopper.step(1.0, 0.8, "epoch_2") is True
