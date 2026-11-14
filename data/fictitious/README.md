# Fictitious data for the local pipeline and the web app

Everything in this folder is invented for training. The programs, the places,
the people, the phone numbers, and the email addresses do not exist. The
names, phone numbers, and emails are planted so the de-identification step has
something to find. The phone numbers are in ranges reserved for fiction and
the email addresses use `example.org`.

There is one focus group in each of three languages. Each has one moderator
and five participants, 26 participant turns, a codebook of five themes, a
roster, and codes for 20 of the 26 passages.

| | Spanish | English | French |
| --- | --- | --- | --- |
| Program | Ahorro en Grupo, San Isidro del Valle | Village Savings Circle, Marlow Creek | Tontine du quartier, Koumbadougou |
| Transcript | `grupo_focal_ficticio.md` | `focus_group_fictitious_en.md` | `groupe_de_discussion_fictif.md` |
| Codebook | `codebook_ficticio.json` | `codebook_fictitious_en.json` | `codebook_fictif.json` |
| Roster | `lista_participantes.txt` | `participant_list_en.txt` | `liste_participants.txt` |
| Codes for 20 passages | `codigos_humanos.csv` | `human_codes_en.csv` | `codes_humains.csv` |
| No-theme label | `Fuera de tema` | `Off topic` | `Hors sujet` |

- The codebooks use the same `"Name: definition"` format as
  `data/themes/help_themes.json`.
- The roster is the list a research team would have. The de-identification
  step uses it to catch names the language model misses. Each transcript also
  names one person who is not on the roster, and one small place.
- The codes file stands in for a person's codes, to check the model codes
  against. The English and French files were written together with their
  transcripts as a teaching sample, not coded independently by a second
  person, so expect the agreement with a model to look better than it would on
  real data.

The command-line local pipeline (`just local-deid` and the steps after it)
uses the Spanish set. The web app offers all three: on the first page, choose
the language of the example and open it.
