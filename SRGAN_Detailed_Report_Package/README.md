# SRGAN Midterm Cats Versus Dogs

**Author:** Owen Klimek, University of North Carolina at Charlotte.

This experiment compares a MobileNetV2 classifier trained on real 128 × 128 RGB images (A) with the same classifier trained on final SRGAN reconstructions (B). The compact generator learns 32 × 32 to 128 × 128 super resolution. Training completed **5 reconstruction warm-up epochs plus 150 full adversarial epochs**.

The detailed methods, all figures, worked calculations, limitations and reproduction procedure are available in [the Word report](SRGAN_Detailed_Report.docx) and [the PDF report](SRGAN_Detailed_Report.pdf).

This documentation package includes the two report formats, the report figures in `figures/`, and selected configuration, metric and history files in `verified_records/`. The full per-image predictions and image manifest are available through the original results-export link below. Add the executed notebook from the completed run to your repository.

## Main result

On identical real test images, A achieved **98.0612% accuracy**, compared with **97.1253%** for B. On identical SRGAN test inputs, B achieved **94.3442%**, compared with **94.0500%** for A. B also performed better on bicubic test inputs. These are separate matched-input comparisons; generated-image training did not improve the primary real-image result.

## Data and split

- Dataset: the same prepared cats-versus-dogs collection used in Assignment 1, supplied as `dogcat_dataset.zip`.
- Prepared images: **24,930**, comprising **12,472 cats** and **12,458 dogs**.
- Labels: cat = 0; dog = 1, the positive class.
- Outer split: 70% training pool and 30% test, stratified with seed 42.
- Inner validation: 10% of the training pool; overall approximately 63% fitting, 7% validation, 30% testing.
- Counts: **15,705 fitting**, **1,746 validation**, **7,479 test**.
- All three fitting stages use the same fitting IDs. Exact pixel duplicates are checked before splitting.
- The midterm preparation pass recorded no corrupt/unknown-label exclusions and no exact duplicates removed from the already prepared Assignment 1 collection.

The dataset identity, split IDs, raw predictions and training records are available in the [verified original results export](https://drive.google.com/file/d/13auqlrQchph1HvrUDm0UfthXpKk4o-df/view?usp=drivesdk) and [experiment folder](https://drive.google.com/drive/folders/1E842D4NiyKNHzssSEiKoVKJuS9qmrL98).

Prepared data assets:

- [RGB cache with image arrays and labels](https://drive.google.com/file/d/1JvcSS0FPpKgNqHfmi1V-dv-t1BA39HYb/view?usp=drivesdk)
- [Image manifest with source paths labels and hashes](https://drive.google.com/file/d/1lDzIyEMt4pNlYQ6v1RehTmM_gnzGyRcG/view?usp=drivesdk)
- [Exact split indices](https://drive.google.com/file/d/1IK9-kNZR4rA4BNcUjhtmQM5L1Y52004k/view?usp=drivesdk)
- [Prepared-data metadata](https://drive.google.com/file/d/1B8gXlpIHVgNYr2WHcvfKJpMWWTvryw6p/view?usp=drivesdk)

If the original archive is unavailable, Appendix A of the report provides tested code to reconstruct the exact prepared RGB targets as lossless PNG images from the cache and manifest. This reconstructs the 128 × 128 targets used here, not the earlier raw photographs. Verify the fingerprint before training:

```text
01a704750e08bc899b4388540ea60e9d7fd34b5fcf9336fc5faf4158b298da67
```

Verify that the evaluator can access the linked Drive assets. Existing sharing settings were not changed when preparing this documentation.

## Reproduce the run

1. Add the **executed** `SRGAN_Midterm_Colab.ipynb` from the completed run to this repository. The report package does not automatically capture the Colab execution session.
2. Open that notebook in Google Colab and select an available GPU runtime. The recorded GPU was a Tesla T4.
3. Mount Drive, set `DATA_SOURCE` to the actual Assignment 1 ZIP or prepared-image folder, and set a writable `DRIVE_BASE`. For an independent reproduction choose a new experiment name. For resuming keep the original name and fixed settings.
4. Use `experiment_spec.json` and the environment record. Prepare RGB images and verify counts, labels and the fingerprint. Reuse `split_indices.json` for identical membership.
5. Show resized and augmented examples. Train A for 5 frozen-backbone head epochs at 0.001 and 3 fine-tuning epochs at 0.00001. Only the last 30 backbone layers are eligible for fine-tuning; batch-normalization layers stay frozen.
6. Train G for 5 warm-up epochs, then train G and D for **150 additional full epochs** using all fitting images. Use Adam at 0.0001 and batch size 32. Checkpoints save each completed epoch.
7. Generate the final SR reconstructions once, retain source labels and split IDs, round to uint8 and cache them. Initialize B from the same saved initial classifier as A. Train B with the same classifier schedule, using generated fitting and validation images.
8. Evaluate both classifiers on the identical real, bicubic and SRGAN test inputs. Use threshold 0.5 for hard predictions and probabilities for ROC AUC. Compute SRGAN/bicubic PSNR and SSIM against real targets.
9. Save the executed notebook, results, report, and model download links. Numerical results can vary across independent GPU runs despite fixed seeds.

## Implementation details

The generator has 8 residual blocks with 32 feature channels and two pixel-shuffle enlargements. The discriminator uses 8 convolutional stages, global average pooling, a 256-unit dense layer and a real/fake logit. Frozen ImageNet VGG19 `block3_conv4` features supply content loss. This is a compact SRGAN adaptation, not an exact reproduction of every setting in the original paper.

```text
Generator loss = mean absolute pixel error
               + 0.01 × MSE(VGG(SR)/12.75, VGG(HR)/12.75)
               + 0.001 × BCE(real target 1, D(SR))

Discriminator loss = 0.5 × [BCE(real target 0.9, D(HR))
                          + BCE(fake target 0, D(detached SR))]
```

Images enter datasets as float32 in [0,1]. MobileNetV2 applies `2*x - 1` internally; VGG19 uses its own pretrained preprocessing after multiplication by 255. Classifiers use training-only horizontal flips, small rotations, zoom and contrast. GAN pairs share their training flip before LR generation. Validation and test transformations are deterministic.

## Measured classifier results

| Model | Test inputs | Accuracy % | F1 dog | ROC AUC | Precision dog | Recall dog |
|---|---|---:|---:|---:|---:|---:|
| A | real | 98.0612 | 0.980592 | 0.997993 | 0.980986 | 0.980198 |
| B | real | 97.1253 | 0.971774 | 0.997754 | 0.953866 | 0.990367 |
| A | bicubic | 87.9262 | 0.868578 | 0.962237 | 0.952138 | 0.798501 |
| B | bicubic | 89.8516 | 0.900197 | 0.969225 | 0.884953 | 0.915975 |
| A | srgan | 94.0500 | 0.940051 | 0.986829 | 0.946555 | 0.933637 |
| B | srgan | 94.3442 | 0.944393 | 0.989525 | 0.928165 | 0.961199 |

All rows use the same 7,479 held-out IDs. Accuracy is a percentage; the other columns are fractions. F1, precision and recall refer to the dog class. Independent verification recalculated all six rows from the saved prediction CSVs and checked IDs and labels against the manifest and split.

## Reconstruction results

| Method | Mean test PSNR dB | Mean test SSIM |
|---|---:|---:|
| Bicubic | 25.063898 | 0.707662 |
| SRGAN | 25.037427 | 0.694663 |

Metrics are averaged per image in RGB, with maximum value 1, no border crop, and PSNR capped at 100 dB for exact matches. Image-quality metrics use floating-point G outputs; SR classification uses the rounded uint8 cache. The SRGAN pipeline improved classification over bicubic inputs for both classifiers, while its aggregate reconstruction fidelity remained slightly lower. Generated texture must not be treated as guaranteed recovery of true detail.

## Checkpoint recovery

Keep the complete experiment folder, `experiment_spec.json`, `split_indices.json`, caches and checkpoint companion files. After a disconnect, select a GPU, mount the same Drive and run notebook sections in order. The updated split cell restores saved settings and exact IDs. G/D checkpoints restore both networks, both optimizers, loss scales and epoch counters. A/B stages also preserve optimizer and epoch state. With a one-epoch save interval, unfinished work since the latest saved epoch may need to be repeated.

Do not overwrite metadata to bypass a different dataset or precision policy. Completed classifiers are loaded and completed GAN stages are skipped. A session budget is a best-effort boundary check; runtime resources are not guaranteed.

## Recorded environment and time

Python 3.13.16; TensorFlow 2.21.0; Keras 3.13.2; NumPy 2.1.3; Pillow 11.3.0; scikit-learn 1.6.1; Matplotlib 3.10.0; ReportLab 5.1.0; Tesla T4; `mixed_float16`.

The saved stage intervals sum to **19,465.34 seconds**, approximately **5 hours 24 minutes 25 seconds**. This is not end-to-end wall-clock duration. GAN intervals exclude post-epoch validation, checkpointing and preview plots; classifier intervals include fit, validation and checkpoint operations. Data preparation, generation, final testing, exports and time between sessions add elapsed time.

## Final models

- [Classifier A](https://drive.google.com/file/d/1qC8aXpvf5aD4BZxbyL1nnhukF-OVTIOk/view?usp=drivesdk)
- [Classifier B](https://drive.google.com/file/d/1A4qNbn_Z441fFWToxnTLXtRJpHFo-pDr/view?usp=drivesdk)
- [Final generator](https://drive.google.com/file/d/1Ec1PnorC_nmVgmzVhQbW86VVD5Dyxh_-/view?usp=drivesdk)
- [Final discriminator](https://drive.google.com/file/d/1JnbR-KuvNAvAEtPuv2JMQktHusdYQcLu/view?usp=drivesdk)

The generator uses a registered `PixelShuffle` layer. Load its class definition from the notebook before loading its `.keras` file. Large models, datasets and checkpoints are linked rather than placed in the small documentation package.

## Submission

Include the executed notebook, this README, either report format, and exported metrics/figures in your GitHub repository. Provide accessible data/model assets. Submit **only the repository URL** on Canvas under **Assignment/Mid-Term Exam**; do not upload files to Canvas.

## References

- Ledig et al., *Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network*, CVPR 2017. https://arxiv.org/abs/1609.04802
- Keras MobileNetV2. https://keras.io/api/applications/mobilenet/mobilenet_models/
- Keras VGG preprocessing. https://keras.io/api/applications/vgg/vgg_preprocessing/
- TensorFlow training checkpoints. https://www.tensorflow.org/guide/checkpoint
- scikit-learn ROC AUC. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
