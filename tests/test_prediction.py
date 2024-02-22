from __future__ import annotations

import os
import pyapetnet
import tensorflow as tf

def test_minimal_prediction() -> None:

    model_path = os.path.join(os.path.dirname(pyapetnet.__file__), "trained_models")
    model_name = "200824_mae_osem_psf_bet_10"

    co = {"ssim_3d_loss": None, "mix_ssim_3d_mae_loss": None}
    model = tf.keras.models.load_model(
        os.path.join(model_path, model_name), custom_objects=co
    )

    x = [tf.ones((1,15,15,15,1)), tf.ones((1,15,15,15,1))]

    pred = model.predict(x)
