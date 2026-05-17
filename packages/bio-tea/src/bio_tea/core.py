# Copyright 2026 tznurmin
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import random

from .augment import windowing
from .augment.pipeline import augment_tokens
from .augment.words import extract_span_phrases
from .resources import species as species_resources
from .transforms.scramble import scramble_text
from .transforms.species_switch import switch_species


class TEA:

    def __init__(
        self,
        tokenizer,
        rseed=None,
        non_stops=None,
        max_len=480,
        max_final_len=505,
        extra_species=None,
        reserved_strains=None,
    ):
        # Instance RNG: avoids global random contamination and makes rseed=0 work.
        self.r_seed = rseed
        self.rng = random.Random(rseed)

        self.tokenizer = tokenizer
        # Token counting must work for any tokenizer compatible with TEA.
        # Primary tokenizer interface: .tokenize(str)->list[str].
        # Fallback: whitespace word count.
        tok_fn = getattr(tokenizer, "tokenize", None)
        if callable(tok_fn):
            def _count_with_tokenizer(s):
                try:
                    toks = tok_fn(s, verbose=False)
                except TypeError:
                    toks = tok_fn(s)
                return len(toks)

            self._token_counter = _count_with_tokenizer
        else:
            self._token_counter = lambda s: len(s.split())

        self.maxlen = max_len
        self.max_final_len = max_final_len
        self.reserved_strains = set(reserved_strains or [])

        self.non_stops = non_stops or ['fig.', 'al.', 'sp.', 'spp.', 'e.g.', 'pv.', 'eg.']

        try:
            species_text = species_resources.load_species_text()
        except FileNotFoundError as e:
            raise FileNotFoundError("species.txt not found") from e

        self.all_species = species_resources.build_all_species(species_text)
        for sp in (extra_species or []):
            s = str(sp).strip()
            if not s:
                continue
            self.all_species.add(s)
            parts = s.split(' ')
            if len(parts) >= 2 and parts[0]:
                self.all_species.add(f"{parts[0][0]}. {parts[1]}")
        self.species_list = species_resources.build_species_list(self.all_species, rng=self.rng)

    def extract_sentence(self, sp, text):
        return windowing.extract_sentence(sp, text, self.non_stops)

    def num_tokens(self, text, tokenizer=None):
        # The optional tokenizer argument is accepted but not used.
        return self._token_counter(text)

    def maximise(self, loc, text):

        return windowing.maximise(
            loc=loc,
            tokens=text,
            maxlen=self.maxlen,
            token_counter=self._token_counter,
            non_stops=self.non_stops,
            rng=self.rng,
        )

    def is_stop(self, word: str):

        return windowing.is_stop(word, self.non_stops)

    def _regenerate_list(self):
        self.species_list = species_resources.build_species_list(self.all_species, rng=self.rng)
    def switch(self, text: str, species_pool=None):
        """Switch species mentions in text.

        If species_pool is provided, replacements are sampled from that list
        (full binomials). Sampling remains deterministic under the instance RNG
        when call order is stable.
        """

        if species_pool is None:
            sample = self.sample
        else:
            pool = list(species_pool)
            if not pool:
                raise ValueError("species_pool is empty")

            def sample():
                if not pool:
                    pool.extend(species_pool)
                if not pool:
                    raise ValueError("species_pool is empty")
                i = self.rng.randrange(len(pool))
                return pool.pop(i)

        return switch_species(text, all_species=self.all_species, sample=sample)

    def scramble(
        self,
        text,
        wordlist,
        force_diff=False,
        skipped_chars=None,
        conserved=None,
        reserved_phrases=None,
        return_mapping=False,
    ):
        return scramble_text(
            text,
            wordlist,
            force_diff=force_diff,
            skipped_chars=skipped_chars,
            conserved=conserved,
            reserved_phrases=set(self.reserved_strains if reserved_phrases is None else reserved_phrases),
            return_mapping=return_mapping,
            rng=self.rng,
        )

    def sample(self):
        if len(self.species_list) == 0:
            self._regenerate_list()
        return self.species_list.pop()

    def _extract_words(self, text, curation_data):
        return extract_span_phrases(text, curation_data)

    def augment(self, text: str, curation_data: dict, scramble=None, l2l=None) -> str:
        '''
        Pool labels with l2l. By default, l2l preserves all labels.
        When l2l is provided, labels not present in the mapping are assigned 'O'.
        '''

        return augment_tokens(
            tokens=text,
            curation_data=curation_data,
            maximise=lambda loc, toks: self.maximise(loc, toks),
            switch=self.switch,
            scramble=self.scramble,
            num_tokens=lambda s: self.num_tokens(s, self.tokenizer),
            max_final_len=self.max_final_len,
            scramble_tags=scramble or [],
            l2l=l2l,
        )
