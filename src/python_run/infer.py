import logging
import time
import wave
from pathlib import Path
from typing import Any, Dict, Optional

from piper import PiperVoice  # Adjust the import as needed for your project structure
from piper.download import ensure_voice_exists, find_voice, get_voices
from pathlib import Path

class PiperSynthesizer:
    def __init__(
        self,
        model: str,
        config: Optional[str] = None,
        data_dir: Optional[str] = None,
        download_dir: Optional[str] = None,
        use_cuda: bool = False,
        update_voices: bool = False,
        debug: bool = False,
    ):
        self._LOGGER = logging.getLogger(__name__)
        logging.basicConfig(level=logging.DEBUG if debug else logging.INFO)

        if not data_dir:
            data_dir = str(Path.cwd())
        if not download_dir:
            download_dir = data_dir

        model_path = Path(model)
        if not model_path.exists():
            voices_info = get_voices(download_dir, update_voices=update_voices)
            aliases_info: Dict[str, Any] = {}
            for voice_info in voices_info.values():
                for voice_alias in voice_info.get("aliases", []):
                    aliases_info[voice_alias] = {"_is_alias": True, **voice_info}

            voices_info.update(aliases_info)
            ensure_voice_exists(model, [data_dir], download_dir, voices_info)
            model, config = find_voice(model, [data_dir])

        self.voice = PiperVoice.load(model, config_path=config, use_cuda=use_cuda)
        self.config = config

    def synthesize_speech(
        self,
        text: str,
        output_file: Optional[str] = None,
        output_dir: Optional[str] = None,
        speaker: Optional[int] = 0,
        length_scale: Optional[float] = None,
        noise_scale: Optional[float] = None,
        noise_w: Optional[float] = None,
        sentence_silence: float = 0.0,
    ) -> Optional[str]:
        synthesize_args = {
            "speaker_id": speaker,
            "length_scale": length_scale,
            "noise_scale": noise_scale,
            "noise_w": noise_w,
            "sentence_silence": sentence_silence,
        }

        self._LOGGER.debug(f"Input text length: {len(text)}")
        output_file_path = None

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            wav_path = output_dir / f"{time.monotonic_ns()}.wav"
            with wave.open(str(wav_path), "wb") as wav_file:
                self.voice.synthesize(text, wav_file, **synthesize_args)
            self._LOGGER.info("Wrote %s", wav_path)
            output_file_path = str(wav_path)
        else:
            if (not output_file) or (output_file == "-"):
                with wave.open(sys.stdout.buffer, "wb") as wav_file:
                    self.voice.synthesize(text, wav_file, **synthesize_args)
            else:
                with wave.open(output_file, "wb") as wav_file:
                    self.voice.synthesize(text, wav_file, **synthesize_args)
                output_file_path = output_file

        return output_file_path


if __name__ == "__main__":
    # Example usage
    text = "Net financial savings of Indian households is at a 50-year low  That one headline sent everyone into a tizzy last month See  the net financial savings is simply the difference between the investments made in financial assets such as bank deposits and the loans taken. So if this band calculated as percentage of GDP is narrowing it could indicate that people are earning less they’re saving less money and taking loans to fund their consumption And that’s quite a scary phenomenon. Because we all know what happens when debt gets out of hand. People begin to default. They might enter into a debt trap where they borrow just to pay back old loans. And their long-term financial well being gets compromised. Meanwhile banks get saddled with bad loans. That affects their risk-taking ability too. They might dial back on lending money to parts of the economy that really need it.It could be a vicious cycle.But wait…the government was quick to provide a rebuttal. It pointed out that while people’s financial savings might be lower, they were actually investing more in physical assets such as housing. And even SBI economic research team jumped in echoing that same sentiment. They said that while loans were rising, a substantial chunk was for buying homes 50% of the loan value in the system as of March 2023 was for homes. And that meant that physical assets were being built by households too. So yeah maybe things are not so dire Also, there might be another positive lens through which we can view all this. It’s something that Marcellus, an investment management firm pointed out and its called the Lifecycle Hypothesis Model. It was created in the 1950s by Franco Modigliani and Richard Brumberg. And in a nutshell it says that when people are young, their income might be on the lower side. So they have a tendency to borrow money to spend on consumption. YOLO and all that, right!This edition, like the earlier ones, is divided into two parts: Part 1, Relativity and Quantum Mechanics: The Foundation of Modern Physics, and Part 2, Applications We continue to open Part 1 with the two relativity chapters. This location for relativity is firmly endorsed by users and reviewers. The rationale is that this arrangement avoids separation of the foundations of quantum mechanics in Chapters 3 through 8 from its applications in Chapters 9 through 12. The two-chapter format for relativity provides instructors with the flexibility to cover only the basic concepts or to go deeper into the subject. Chapter 1 covers the essentials of special relativity and includes discussions of several paradoxes, such as the twin paradox and the pole-in-the-barn paradox, that never fail to excite student interest. Relativistic energy and momentum are covered in Chapter 2, which concludes with a mostly qualitative section on general relativity that emphasizes experimental tests. Because the relation is the result most needed for the later applications chapters it is possible to omit Chapter 2 without disturbing continuity. Chapters 1 through 8 have been updated with a number of improved explanations and new diagrams. Several classical foundation topics in those chapters have been moved to the Classical Concept Review or recast as MORE sections. Many quantitative topics are included as MORE sections on the Web site. Examples of these are the derivation of Comptons equation (Chapter 3), the details of Rutherfords alpha-scattering theory (Chapter 4), the graphical solution of the finite square well (Chapter 6), and the excited states and spectra of two-electron atoms (Chapter 7). The comparisons of classical and quantum statistics are illustrated with several examples in Chapter 8 and unlike the other chapters in Part 1, Chapter 8 is arranged to be covered briefly and qualitatively if desired. This chapter like Chapter 2 is not essential to the understanding of the applications chapters of Part 2 and may be used as an applications chapter or omitted without. A couple of days ago, the Financial Times released a controversial report on the Adani Group. It alleged that the conglomerate has been inflating fuel costs, forcing millions of Indian consumers and businesses to overpay for electricity. How’s that you ask Well FT suggests that the group has repeatedly imported coal from countries like Indonesia at inflated prices. Sometimes exceeding even 50-100% of the value shown on paper while it was being exported from the source country. Now, the conglomerate obviously doesn’t agree with these findings. But we wrote a story about what may be happening after all. We couldnt publish it due to an unforeseen technical error. But you can listen to it on our daily podcast here Spotify here apple Podcasts and here Google Podcasts. Bank of Baroda is involved in fraud In July, international media channel Al Jazeera made a shocking revelation about Bank of Baroda. It said that Indias second largest government-owned bank linked mobile numbers of strangers to its mobile app bob World to boost app registrations, compromising security. The bank denied this report then. But a couple of days ago, the RBI Reserve Bank of India suspended Bank of Baroda from onboarding new clients to its app. So what really happened? We took a look at it in our Tuesday newsletter here."
    model_path = "/vidgen/sourabh/piper/src/python_run/checkpoint/en/en_US/lessac/medium/en_US-lessac-medium.onnx"
    config_path = "/vidgen/sourabh/piper/src/python_run/checkpoint/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"
    output_path = "/vidgen/sourabh/piper/src/python_run/output/output_4.wav"

    start = time.time()
    synthesizer = PiperSynthesizer(
        model=model_path,
        config=config_path,
        use_cuda=True,
        debug=True
    )

    output_file = synthesizer.synthesize_speech(
        text=text,
        output_file=output_path,
        speaker=0,
        length_scale=1.0,
        noise_scale=0.667,
        noise_w=0.8,
        sentence_silence=0.0,
    )

    end = time.time()
    print(f"Time taken: {end - start} seconds")
    print(f"Generated audio file saved at: {output_file}")
