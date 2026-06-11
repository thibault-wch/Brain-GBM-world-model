# coding=utf-8
import os
import csv
import collections
import re
from typing import List, Dict, Any, Optional, Sequence, Tuple

import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from monai import transforms as mtransforms

IGNORE_INDEX = -100

# ----- Hardcoded CSV list -----
DEFAULT_CSVS = [
    "/root/autodl-tmp/dataset/csv_files/LUMIERE.csv",
    "/root/autodl-tmp/dataset/csv_files/MU.csv",
    "/root/autodl-tmp/dataset/csv_files/UCSF.csv",
]

EXTERNAL_CSVs = [
    "/root/autodl-tmp/dataset/csv_files/RHUH_v2.csv",
    "/root/autodl-tmp/dataset/csv_files/UCSD_v3.csv",
]

REQUIRED_COLS = {"patient_id", "t1_time_id", "t2_time_id", "template_text_en", "conducted_treatment"}


def _dedup_test_caption_by_tid1(anno: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Deduplicate test and captioning records based on t1_time_id.
    Keys used: site + normalized_tid1 + patient_id.
    """
    seen = set()
    kept = []
    for r in anno:
        site = (r.get("__site__", "") or "").upper()
        pid = (r.get("patient_id", "") or "").strip()
        tid1_raw = (r.get("t1_time_id", "") or "").strip()
        tid1 = _normalize_time_id(site, tid1_raw)

        key = (site, pid, tid1)

        if key in seen:
            continue
        seen.add(key)

        r_copy = r.copy()
        r_copy["t1_time_id"] = tid1
        kept.append(r_copy)

    return kept


def _read_and_validate_pair_csvs(csv_paths: List[str], external_csv_paths: List[str], mode: str = "train") -> List[
    Dict[str, str]]:
    """
    Read CSVs, enforce required columns, inject __site__ tag, and filter by train/test splits.
    """
    assert mode in {"train", "test", "external"}, "Mode must be 'train', 'test', or 'external'."

    dfs = []
    if mode in {'train', 'test'}:
        present = [p for p in csv_paths if p and os.path.isfile(p)]
        if not present:
            return []

        test_csv = pd.read_csv('/root/autodl-tmp/dataset/csv_files/test_information.csv', dtype=str)
        test_patient_id = set(test_csv["patient_id"].astype(str).str.strip())

        for pth in present:
            site = os.path.splitext(os.path.basename(pth))[0].upper()
            df = pd.read_csv(pth, dtype=str, encoding="utf-8")
            df.columns = df.columns.str.strip()

            missing = REQUIRED_COLS - set(df.columns)
            if missing:
                raise ValueError(f"Missing required columns in {pth}: {sorted(missing)}")

            cols = [c for c in REQUIRED_COLS if c in df.columns]
            df = df[cols].copy()

            # Clean string values
            for c in cols:
                df[c] = df[c].apply(lambda x: str(x).strip() if pd.notna(x) else "")

            df["__site__"] = site
            dfs.append(df)

        if not dfs:
            return []

        big = pd.concat(dfs, ignore_index=True)
        in_test = big["patient_id"].isin(test_patient_id)
        big = big.loc[~in_test] if mode == "train" else big.loc[in_test]

    else:
        present = [p for p in external_csv_paths if p and os.path.isfile(p)]
        if not present:
            return []

        for pth in present:
            base = os.path.splitext(os.path.basename(pth))[0]
            site = base.split("_", 1)[0].upper()

            df = pd.read_csv(pth, dtype=str, encoding="utf-8")
            df.columns = df.columns.str.strip()

            missing = REQUIRED_COLS - set(df.columns)
            if missing:
                raise ValueError(f"Missing required columns in {pth}: {sorted(missing)}")

            cols = [c for c in REQUIRED_COLS if c in df.columns]
            df = df[cols].copy()

            # Clean string values
            for c in cols:
                df[c] = df[c].apply(lambda x: str(x).strip() if pd.notna(x) else "")

            df["__site__"] = site
            dfs.append(df)

        if not dfs:
            return []

        big = pd.concat(dfs, ignore_index=True)

    return big.astype(str).to_dict(orient="records")


def _normalize_time_id(site: str, tid: str) -> str:
    site = (site or "").upper()
    if site in {"UCSF", "MU"}:
        if tid.startswith("Timepoint") or tid.startswith("TIMEPOINT"):
            return tid
        return f"Timepoint_{tid}"
    if site in {"UCSD", "UCSD_V2"}:
        if "_" in tid:
            suf = tid[tid.find("_"):]
            return f"Timepoint{suf}"
        else:
            return f"Timepoint_{tid}"
    return tid


def format_sequence_gen_qwen2_5(text_tokens, system_tokens, bos_id, eos_id, boi_id, eoi_id, pad_id, img_pad_id,
                                num_image_tokens, max_seq_len, system_token_len):
    if system_token_len == 0:
        modality_positions = torch.tensor([[len(text_tokens) + 1 + 1, num_image_tokens]])
        text_labels = [-100] + [-100] * len(text_tokens) + [-100] + [-100] * num_image_tokens + [-100] + [-100]
        text_tokens = [bos_id] + text_tokens + [boi_id] + [img_pad_id] * num_image_tokens + [eoi_id] + [eos_id]
    else:
        modality_positions = torch.tensor([[1 + system_token_len + len(text_tokens) + 1 + 1, num_image_tokens]])
        text_labels = (
                [bos_id] + [-100] * len(system_tokens[0] + system_tokens[1] + text_tokens) + [eos_id] +
                [-100] * len(system_tokens[2]) +
                [boi_id] + [-100] * num_image_tokens + [eoi_id] + [eos_id]
        )
        text_tokens = (
                [bos_id] + system_tokens[0] + system_tokens[1] + text_tokens + [eos_id] + system_tokens[2] +
                [boi_id] + [img_pad_id] * num_image_tokens + [eoi_id] + [eos_id]
        )

    text_labels = text_labels + [-100] * (max_seq_len - len(text_labels))
    text_tokens = text_tokens + [pad_id] * (max_seq_len - len(text_tokens))
    text_tokens = torch.tensor(text_tokens)
    text_labels = torch.tensor(text_labels)

    text_mask = torch.where((text_tokens != img_pad_id) & (text_tokens != pad_id),
                            torch.ones_like(text_tokens), torch.zeros_like(text_tokens))
    image_mask = torch.where(text_tokens == img_pad_id,
                             torch.ones_like(text_tokens), torch.zeros_like(text_tokens))

    return text_tokens, text_labels, modality_positions, text_mask, image_mask


def format_sequence_und_masked(prompt_tokens, answer_tokens, bos_id, eos_id, boi_id, eoi_id, pad_id, img_pad_id,
                               num_image_tokens, max_seq_len):
    """
    Understanding (MMU) with ONLY the answer token(s) supervised.
    Layout: [bos, boi, <img_pad>*N, eoi, prompt_tokens + answer_tokens, eos, pad...]
    """
    modality_positions = torch.tensor([[1 + 1, num_image_tokens]])

    # Avoid mutating the original token list reference
    adjusted_answer_tokens = [answer_tokens[0] - 151669] + answer_tokens[1:]

    text_tokens = [bos_id, boi_id] + [img_pad_id] * num_image_tokens + [
        eoi_id] + prompt_tokens + adjusted_answer_tokens + [eos_id]

    text_labels = (
            [-100] +
            [-100] +
            [-100] * num_image_tokens +
            [-100] +
            [-100] * len(prompt_tokens) +
            adjusted_answer_tokens +
            [-100]
    )

    text_labels = text_labels + [-100] * (max_seq_len - len(text_labels))
    text_tokens = text_tokens + [pad_id] * (max_seq_len - len(text_tokens))

    text_tokens = torch.tensor(text_tokens)
    text_labels = torch.tensor(text_labels)

    text_mask = torch.where((text_tokens != img_pad_id) & (text_tokens != pad_id),
                            torch.ones_like(text_tokens), torch.zeros_like(text_tokens))
    image_mask = torch.where(text_tokens == img_pad_id,
                             torch.ones_like(text_tokens), torch.zeros_like(text_tokens))

    return text_tokens, text_labels, modality_positions, text_mask, image_mask


def format_sequence_und_infer(prompt_tokens, answer_tokens, bos_id, eos_id, boi_id, eoi_id, pad_id, img_pad_id,
                              num_image_tokens, max_seq_len):
    """
    Understanding (MMU) inference sequence builder.
    """
    modality_positions = torch.tensor([[1 + 1, num_image_tokens]])

    text_tokens = [bos_id, boi_id] + [img_pad_id] * num_image_tokens + [eoi_id] + prompt_tokens

    text_labels = (
            [-100] +
            [-100] +
            [-100] * num_image_tokens +
            [-100] +
            [-100] * len(prompt_tokens)
    )

    # Note: Tokens modified purely for local use without mutating global state
    _ = [answer_tokens[0] - 151669] + answer_tokens[1:]

    text_tokens = torch.tensor(text_tokens)
    text_labels = torch.tensor(text_labels)

    text_mask = torch.where((text_tokens != img_pad_id) & (text_tokens != pad_id),
                            torch.ones_like(text_tokens), torch.zeros_like(text_tokens))
    image_mask = torch.where(text_tokens == img_pad_id,
                             torch.ones_like(text_tokens), torch.zeros_like(text_tokens))

    return text_tokens, text_labels, modality_positions, text_mask, image_mask


# ---- Conducted treatment dictionary mapping ----
CLASS2TOKEN_MAP = {
    "CRT": "<CRT>",
    "RT": "<RT>",
    "TMZ": "<TMZ>",
    "surgery": "<SURGERY>",
    "active monitoring": "<AM>",
}

_CANON_PATTERNS = [
    (re.compile(r"\bchemoradiotherapy\b|\bchemo\s*\+\s*rt\b|\bcrt\b|\bcmt\b", re.I), "CRT"),
    (re.compile(r"\bradiotherapy\b|\brt\b", re.I), "RT"),
    (re.compile(r"\btemozolomide\b|\btmz\b", re.I), "TMZ"),
    (re.compile(r"\bsurgery\b|\bresection\b|\boperation\b|\bop\b", re.I), "surgery"),
    (re.compile(r"\bactive\s*monitoring\b|\bwatchful\s*waiting\b|\bno\s*treatment\b", re.I), "active monitoring"),
]


def to_special_token(text: str) -> str:
    if not text:
        return CLASS2TOKEN_MAP["active monitoring"]
    t = text.strip()
    if t in CLASS2TOKEN_MAP:
        return CLASS2TOKEN_MAP[t]
    tl = t.lower()
    for pat, cls in _CANON_PATTERNS:
        if pat.search(tl):
            return CLASS2TOKEN_MAP[cls]
    return CLASS2TOKEN_MAP["active monitoring"]


def replace_with_special_tokens(text: str) -> str:
    """
    Substitutes clinical terminology with special tokens.
    """
    if not isinstance(text, str):
        return text
    result = text
    for pattern, cls in _CANON_PATTERNS:
        token = CLASS2TOKEN_MAP[cls]
        result = pattern.sub(lambda m: token, result)
    return result


class MedicalPairImageTextDataset(Dataset):
    """
    Multimodal Dataset Architecture:
      - mmu (is_captioning=True): Supervised text generation.
      - t2i (is_captioning=False): Image generation conditional setup.
    Directory structure: root/patient_id/time_id/{flair, t1ce, t2, seg_mask}.nii.gz
    """

    def __init__(
            self,
            root: str,
            text_tokenizer: Any,
            showo_token_ids: Dict[str, int],
            spatial_size: Sequence[int] = (224, 224, 152),
            max_seq_len: int = 318 + 576 + 2,
            num_image_tokens: int = 576,
            is_captioning: bool = False,
            cond_dropout_prob: float = 0.0,
            system: Tuple[str, str, str] = ("", "", ""),
            strict_files: bool = True,
            use_seg_mask: bool = False,
            comp_mode=False,
            mode='train'
    ):
        super().__init__()
        self.root = root
        self.is_captioning = is_captioning
        self.text_tokenizer = text_tokenizer

        self.pad_id = self.text_tokenizer.pad_token_id
        self.bos_id = showo_token_ids['bos_id']
        self.eos_id = showo_token_ids['eos_id']
        self.boi_id = showo_token_ids['boi_id']
        self.eoi_id = showo_token_ids['eoi_id']
        self.img_pad_id = showo_token_ids['img_pad_id']

        self.max_seq_len = max_seq_len
        self.num_image_tokens = num_image_tokens
        self.cond_dropout_prob = cond_dropout_prob
        self.data_type = "mmu" if self.is_captioning else "t2i"
        self.strict_files = strict_files
        self.use_seg_mask = use_seg_mask
        self.mode = mode
        self.comp_mode = comp_mode

        # System tokens
        self.system_tokens = self.text_tokenizer(system, add_special_tokens=False).input_ids
        self.system_token_len = sum(len(tokens) for tokens in self.system_tokens)
        if len(self.system_tokens[0]) == 0:
            self.max_text_len = max_seq_len - num_image_tokens - 4
        else:
            self.max_text_len = max_seq_len - num_image_tokens - 4 - self.system_token_len - 1

        img_keys = ["flair1", "t1ce1", "t2w1", "flair2", "t1ce2", "t2w2"]
        if self.use_seg_mask:
            img_keys += ["seg1", "seg2"]

        # MONAI base transform pipeline
        self.img_t = mtransforms.Compose([
            mtransforms.LoadImaged(keys=img_keys, image_only=True, ensure_channel_first=True),
            mtransforms.SqueezeDimd(keys=img_keys),
            mtransforms.Spacingd(keys=["flair1", "t1ce1", "t2w1", "flair2", "t1ce2", "t2w2"], pixdim=(1.0, 1.0, 1.0),
                                 mode=("bilinear",) * 6),
            mtransforms.Spacingd(keys=["seg1", "seg2"], pixdim=(1.0, 1.0, 1.0), mode=("nearest",) * 2,
                                 allow_missing_keys=True),
            mtransforms.EnsureTyped(keys=img_keys),
            mtransforms.ScaleIntensityRangePercentilesd(
                keys=["flair1", "t1ce1", "t2w1", "flair2", "t1ce2", "t2w2"],
                lower=0, upper=99.5, b_min=-1.0, b_max=1.0, clip=True, relative=False),
            mtransforms.SpatialCropd(keys=img_keys, roi_center=(120, 120, 78), roi_size=(192, 192, 140)),
            mtransforms.Resized(keys=["flair1", "t1ce1", "t2w1", "flair2", "t1ce2", "t2w2"], spatial_size=spatial_size,
                                allow_missing_keys=True),
            mtransforms.Resized(keys=["seg1", "seg2"], spatial_size=spatial_size, mode='nearest',
                                allow_missing_keys=True),
        ])

        def _zoom_like_crop(keys, use_seg):
            modes = ("bilinear",) * 6 + (("nearest", "nearest") if use_seg else ())
            try:
                return mtransforms.RandZoomd(
                    keys=keys, prob=1.0, min_zoom=1.0, max_zoom=1.5, mode=modes, keep_size=True
                )
            except TypeError:
                return mtransforms.RandZoomd(
                    keys=keys, prob=1.0, zoom_range=(1.0, 1.5), mode=modes, keep_size=True
                )

        self.random_transform = mtransforms.OneOf([
            mtransforms.Compose([]),
            mtransforms.RandFlipd(keys=img_keys, prob=1.0, spatial_axis=0),
            mtransforms.RandFlipd(keys=img_keys, prob=1.0, spatial_axis=1),
            mtransforms.RandFlipd(keys=img_keys, prob=1.0, spatial_axis=2),
            mtransforms.RandAffined(
                keys=img_keys, prob=1.0,
                translate_range=(24, 24, 5), padding_mode="border",
                mode=("bilinear",) * 6 + (("nearest", "nearest") if self.use_seg_mask else ())
            ),
            _zoom_like_crop(img_keys, self.use_seg_mask),
        ], weights=(0.45, 0.05, 0.05, 0.05, 0.15, 0.25))

        self.anno: List[Dict[str, str]] = _read_and_validate_pair_csvs(DEFAULT_CSVS, EXTERNAL_CSVs, self.mode)

        print(f'Dataset initialized. Length: {len(self.anno)}')

    def __len__(self) -> int:
        return len(self.anno)

    def _remap_seg_by_site(self, seg: torch.Tensor, site: str) -> torch.Tensor:
        """
        Remap segmentation indices based on specific site protocols.
        """
        site = (site or "").upper()
        seg_new = seg.clone()
        if site == "LUMIERE":
            m1 = seg == 1
            m2 = seg == 2
            seg_new[m1] = 2
            seg_new[m2] = 1
        elif site in {"UCSF", "UCSD"}:
            seg_new[seg == 4] = 0

        return seg_new

    def _stack_modalities(self, flair1: str, t1ce1: str, t2w1: str, flair2: str, t1ce2: str, t2w2: str,
                          seg1: Optional[str] = None, seg2: Optional[str] = None, mode='train') -> Tuple[
        torch.Tensor, ...]:

        paths_dict = {
            'flair1': flair1, 't1ce1': t1ce1, 't2w1': t2w1,
            'flair2': flair2, 't1ce2': t1ce2, 't2w2': t2w2
        }

        if seg1 is not None and seg2 is not None:
            paths_dict.update({'seg1': seg1, 'seg2': seg2})

        if mode == 'train':
            dict_img = self.random_transform(self.img_t(paths_dict))
        else:
            dict_img = self.img_t(paths_dict)

        img_t1 = torch.cat([
            dict_img['flair1'].permute(0, 3, 1, 2),
            dict_img['t1ce1'].permute(0, 3, 1, 2),
            dict_img['t2w1'].permute(0, 3, 1, 2)
        ], dim=0)

        img_t2 = torch.cat([
            dict_img['flair2'].permute(0, 3, 1, 2),
            dict_img['t1ce2'].permute(0, 3, 1, 2),
            dict_img['t2w2'].permute(0, 3, 1, 2)
        ], dim=0)

        if seg1 is not None and seg2 is not None:
            return img_t1, dict_img['seg1'].permute(0, 3, 1, 2), img_t2, dict_img['seg2'].permute(0, 3, 1, 2)

        return img_t1, img_t2

    def __getitem__(self, idx: int, _retry_count: int = 0) -> Dict[str, Any]:
        """
        Loads the longitudinal image pairs and applies formatting logic. 
        Limits retry count to avoid infinite recursion on bad data runs.
        """
        if _retry_count > 10:
            raise RuntimeError(f"Failed to load data after 10 retries. Last failed idx: {idx}")

        try:
            r = self.anno[idx]
            pid, tid1, tid2 = r["patient_id"], r["t1_time_id"], r["t2_time_id"]
            site = r.get("__site__", "")

            tid1 = _normalize_time_id(site, tid1)
            tid2 = _normalize_time_id(site, tid2)

            dir1 = os.path.join(self.root, site, pid, tid1)
            dir2 = os.path.join(self.root, site, pid, tid2)

            flair1 = os.path.join(dir1, "flair.nii.gz")
            t1ce1 = os.path.join(dir1, "t1c.nii.gz")
            t2w1 = os.path.join(dir1, "t2.nii.gz")
            seg1 = os.path.join(dir1, "seg_mask.nii.gz")

            flair2 = os.path.join(dir2, "flair.nii.gz")
            t1ce2 = os.path.join(dir2, "t1c.nii.gz")
            t2w2 = os.path.join(dir2, "t2.nii.gz")
            seg2 = os.path.join(dir2, "seg_mask.nii.gz")

            need = [flair1, t1ce1, t2w1, flair2, t1ce2, t2w2]
            if self.use_seg_mask:
                need += [seg1, seg2]

            if self.strict_files:
                for pth in need:
                    if not os.path.isfile(pth):
                        raise FileNotFoundError(f"Missing required file: {pth}")

            if self.use_seg_mask:
                t1_img, t1_seg, t2_img, t2_seg = self._stack_modalities(flair1, t1ce1, t2w1, flair2, t1ce2, t2w2, seg1,
                                                                        seg2, self.mode)
            else:
                t1_img, t2_img = self._stack_modalities(flair1, t1ce1, t2w1, flair2, t1ce2, t2w2, None, None, self.mode)

            # MMU setup
            if self.is_captioning:
                full_tpl = r.get("template_text_en", "") or ""
                match = re.split(r"conducted treatment", full_tpl, flags=re.IGNORECASE)

                context = match[0].strip() if len(match) > 1 else full_tpl.strip()
                question = " What’s the next treatment plan?"
                prompt_text = replace_with_special_tokens((context + question).strip())

                prompt_tokens = self.text_tokenizer(prompt_text, add_special_tokens=False,
                                                    truncation=True, max_length=self.max_text_len).input_ids

                special = to_special_token(r.get("conducted_treatment", ""))
                answer_tokens = self.text_tokenizer(special, add_special_tokens=False).input_ids

                if self.mode == 'train':
                    text_tokens, text_labels, modality_positions, text_mask, image_mask = format_sequence_und_masked(
                        prompt_tokens, answer_tokens, self.bos_id, self.eos_id, self.boi_id, self.eoi_id,
                        self.pad_id, self.img_pad_id, self.num_image_tokens, self.max_seq_len
                    )
                else:
                    text_tokens, text_labels, modality_positions, text_mask, image_mask = format_sequence_und_infer(
                        prompt_tokens, answer_tokens, self.bos_id, self.eos_id, self.boi_id, self.eoi_id,
                        self.pad_id, self.img_pad_id, self.num_image_tokens, self.max_seq_len
                    )

                sample = {
                    "site": site,
                    "pid": pid,
                    "text_tokens": text_tokens,
                    "text_labels": text_labels,
                    "images": t1_img,
                    "images_cond": None,
                    "modality_positions": modality_positions,
                    "text_masks": text_mask,
                    "image_masks": image_mask,
                    "texts": (prompt_text + " " + special).strip() if self.mode == 'train' else [
                        (prompt_text + " ").strip(), special],
                    "data_type": "mmu",
                    "file_id": f"{tid1}_{tid2}"
                }

                if self.use_seg_mask:
                    sample["seg_masks"] = self._remap_seg_by_site(t1_seg, site)

            # T2I setup
            else:
                text = replace_with_special_tokens(r.get("template_text_en", ""))
                txt_tokens = self.text_tokenizer(text, add_special_tokens=False,
                                                 truncation=True, max_length=self.max_text_len).input_ids

                text_tokens, text_labels, modality_positions, text_mask, image_mask = format_sequence_gen_qwen2_5(
                    txt_tokens, self.system_tokens, self.bos_id, self.eos_id,
                    self.boi_id, self.eoi_id, self.pad_id, self.img_pad_id,
                    self.num_image_tokens, self.max_seq_len, self.system_token_len
                )

                sample = {
                    "site": site,
                    "pid": pid,
                    "text_tokens": text_tokens,
                    "text_labels": None,
                    "images_cond": t1_img,
                    "images": t2_img,
                    "modality_positions": modality_positions,
                    "text_masks": text_mask,
                    "image_masks": image_mask,
                    "texts": text,
                    "data_type": "t2i",
                    "file_id": f"{tid1}_{tid2}",
                }

                if self.use_seg_mask:
                    if self.comp_mode:
                        sample["seg_masks"] = self._remap_seg_by_site(t1_seg, site)
                        sample["seg_mask1"] = self._remap_seg_by_site(t1_seg, site)
                        sample["seg_mask2"] = self._remap_seg_by_site(t2_seg, site)
                    else:
                        sample["seg_masks"] = self._remap_seg_by_site(t2_seg, site)

            return sample

        except Exception as e:
            print(
                f"Data loading warning at pid={r.get('patient_id')}, site={r.get('__site__')}: {e}. Retrying next index...")
            return self.__getitem__((idx + 1) % len(self), _retry_count + 1)

    def collate_fn(self, batch):
        batch = [b for b in batch if b is not None]
        if len(batch) == 0:
            return {}

        batched = collections.defaultdict(list)
        for sample in batch:
            for k, v in sample.items():
                batched[k].append(v)

        for k, vlist in list(batched.items()):
            if all(isinstance(x, torch.Tensor) for x in vlist):
                try:
                    batched[k] = torch.stack(vlist, dim=0)
                except RuntimeError:
                    # Dimensionality mismatch fallback
                    batched[k] = vlist

        return dict(batched)


def create_medical_dataloader(
        root: str,
        batch_size: int,
        text_tokenizer: Any,
        showo_token_ids: Dict[str, int],
        spatial_size: Sequence[int] = (192, 192, 128),
        max_seq_len: int = 1024,
        num_image_tokens: int = 576,
        is_captioning: bool = False,
        cond_dropout_prob: float = 0.0,
        num_workers: int = 4,
        shuffle: bool = True,
        use_seg_mask: bool = False,
        accelerator: Optional[Any] = None,
        drop_last: bool = True,
        comp_mode: bool = False,
        mode: str = 'train'
) -> DataLoader:
    ds = MedicalPairImageTextDataset(
        root=root,
        text_tokenizer=text_tokenizer,
        showo_token_ids=showo_token_ids,
        spatial_size=spatial_size,
        max_seq_len=max_seq_len,
        num_image_tokens=num_image_tokens,
        is_captioning=is_captioning,
        cond_dropout_prob=cond_dropout_prob,
        strict_files=True,
        use_seg_mask=use_seg_mask,
        comp_mode=comp_mode,
        mode=mode
    )

    if accelerator is not None and accelerator.num_processes > 1:
        sampler = DistributedSampler(
            ds,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            shuffle=shuffle,
            drop_last=drop_last
        )
        shuffle = False
    else:
        sampler = None

    dataloader = DataLoader(
        ds,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=ds.collate_fn,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last
    )

    return dataloader