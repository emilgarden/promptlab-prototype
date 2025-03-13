# PromptLab - LLM Eksperimenteringsverktøy

PromptLab er et verktøy for å eksperimentere med ulike språkmodeller (LLMs), estimere kostnader og sammenligne resultater. Applikasjonen er bygget med Streamlit og støtter modeller fra OpenAI og Anthropic.

![PromptLab Screenshot](screenshot.png)

## Funksjoner

- **Chat-grensesnitt**: Moderne chat-grensesnitt for interaksjon med språkmodeller
- **Støtte for flere modeller**: Integrasjon med både OpenAI og Anthropic modeller
- **Kostnadsestimering**: Nøyaktig estimering av token-bruk og kostnader før og etter generering
- **Valutakonvertering**: Veksling mellom USD og NOK med automatisk oppdatering av valutakurser
- **Cached input**: Støtte for cached input for modeller som tilbyr dette
- **Systemprompter**: Mulighet for å definere systemprompter for å styre modellens oppførsel
- **Samtalehistorikk**: Lagring av samtalehistorikk med token-telling og kostnadsberegning

## Installasjon

1. Klone repositoriet:
```bash
git clone https://github.com/yourusername/promptlab-prototype.git
cd promptlab-prototype
```

2. Installer avhengigheter:
```bash
pip install -r requirements.txt
```

3. Opprett en `.env` fil med API-nøkler:
```
OPENAI_API_KEY=your-openai-key-here
ANTHROPIC_API_KEY=your-anthropic-key-here
```

4. Start applikasjonen:
```bash
streamlit run app.py
```

## Bruk

1. Velg leverandør og modell i venstre panel
2. Angi systemprompten (valgfritt)
3. Skriv inn din melding i tekstfeltet
4. Se kostnadsestimater i høyre panel
5. Klikk "Send" for å generere svar
6. Se faktisk token-bruk og kostnader etter generering

## Modeller

Applikasjonen støtter følgende modeller:

### OpenAI
- GPT-3.5 Turbo
- GPT-3.5 Turbo 16K
- GPT-4o
- GPT-4o mini

### Anthropic
- Claude 3.7 Sonnet
- Claude 3.5 Sonnet (nyeste)
- Claude 3.5 Sonnet (tidligere)
- Claude 3.5 Haiku
- Claude 3 Opus
- Claude 3 Haiku

## Kostnadsberegning

Applikasjonen beregner kostnader basert på:
- Antall input tokens × pris per million tokens
- Antall output tokens × pris per million tokens

Prisene er oppdatert per november 2024 og kan endres over tid.

## Valutakonvertering

Valutakurser hentes fra Frankfurter API og caches i 4 dager for å redusere API-kall.

## Krav

- Python 3.8+
- Streamlit
- OpenAI Python SDK
- Anthropic Python SDK
- python-dotenv
- requests

## Bidrag

Bidrag til prosjektet er velkomne! Vennligst åpne en issue eller pull request for forslag til forbedringer.

## Lisens

Dette prosjektet er lisensiert under MIT-lisensen - se [LICENSE](LICENSE) filen for detaljer. 