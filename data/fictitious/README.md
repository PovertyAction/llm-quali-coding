# Fictitious data for the local pipeline

Everything in this folder is invented for training. The program (Ahorro en
Grupo), the municipality (San Isidro del Valle), the people, the phone number,
and the email address do not exist. The names, phone number, and email are
planted so the de-identification step has something to find.

- `grupo_focal_ficticio.md`: a short Spanish focus group, one moderator and five
  participants, 26 participant turns.
- `codebook_ficticio.json`: five themes, in the same `"Name: definition"` format
  as `data/themes/help_themes.json`.
- `lista_participantes.txt`: the roster a research team would have. The
  de-identification step uses it to catch names the language model misses.
- `codigos_humanos.csv`: a person's codes for 20 of the 26 passages, used to
  check the model codes. Passages that fit no theme are coded `Fuera de tema`.
