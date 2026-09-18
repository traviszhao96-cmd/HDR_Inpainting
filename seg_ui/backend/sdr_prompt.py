"""Shared SDR removal instructions for the UI backend and saved-image probes."""

NO_NEW_PEOPLE = (
    "Inside the masked region, reconstruct only empty, unoccupied background surfaces. "
    "Do not generate any people, faces, heads, bodies, limbs, human silhouettes or portraits. "
    "Do not replace the removed subject with another person or extend nearby people into the mask. "
    "Preserve existing people outside the mask without adding new ones."
)

DEFAULT_SDR_PROMPT = (
    "Remove all selected subjects and objects inside the mask. Reconstruct the background "
    "as if they had never been there, continuing surrounding surfaces, geometry, textures "
    "and color gradients. Match perspective, lighting and colors. Do not add new objects. "
    + NO_NEW_PEOPLE
)


def removal_prompt(prompt):
    text = (prompt or "").strip() or DEFAULT_SDR_PROMPT
    return text if NO_NEW_PEOPLE in text else text + "\n\n" + NO_NEW_PEOPLE
