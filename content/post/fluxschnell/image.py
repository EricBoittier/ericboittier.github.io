from diffusionkit.mlx import FluxPipeline

pipeline = FluxPipeline(
    shift=1.0,
    model_version="argmaxinc/mlx-FLUX.1-schnell",
    low_memory_mode=True,
    a16=True,
    w16=True,
)

HEIGHT = 512
WIDTH = 512
NUM_STEPS = 2  #  4 for FLUX.1-schnell, 50 for SD3
CFG_WEIGHT = 0.0  # for FLUX.1-schnell, 5. for SD3

image, _ = pipeline.generate_image(
    "a giant hamster with a tiny hat",
    cfg_weight=CFG_WEIGHT,
    num_steps=NUM_STEPS,
    latent_size=(HEIGHT // 8, WIDTH // 8),
)

image.save("image.png")
