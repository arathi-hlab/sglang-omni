use std::fmt;

use serde::de::{DeserializeSeed, IgnoredAny, MapAccess, SeqAccess, Visitor};

use crate::worker_pool::{ReferenceForm, SpeechResponseFormat, SpeechTask};

/// Raw routing fields shared by stateless and WebSocket speech requests.
#[derive(Clone, Default)]
pub(crate) struct SpeechFields {
    pub(crate) model: Option<Option<String>>,
    pub(crate) response_format: Option<Option<String>>,
    pub(crate) stream: Option<Option<bool>>,
    pub(crate) task: Option<Option<String>>,
    pub(crate) voice: Option<Option<String>>,
    voice_present: bool,
    pub(crate) ref_audio: Option<Option<String>>,
    pub(crate) references: Option<Option<ReferenceFlags>>,
}

#[derive(Clone, Copy, Default)]
pub(crate) struct ReferenceFlags {
    pub(crate) list: bool,
    pub(crate) vq_codes: bool,
}

/// Reads one routing field, returning false without consuming unknown values.
pub(crate) fn read_field<'de, A>(
    key: &str,
    map: &mut A,
    fields: &mut SpeechFields,
) -> Result<bool, A::Error>
where
    A: MapAccess<'de>,
{
    match key {
        "model" => fields.model = Some(map.next_value()?),
        "response_format" => fields.response_format = Some(map.next_value()?),
        "task_type" => fields.task = Some(map.next_value()?),
        "voice" => {
            fields.voice = Some(map.next_value()?);
            fields.voice_present = true;
        }
        "speaker" => {
            let value = map.next_value()?;
            if !fields.voice_present {
                fields.voice = Some(value);
            }
        }
        "ref_audio" => fields.ref_audio = Some(map.next_value()?),
        "references" => fields.references = Some(map.next_value_seed(NullableReferencesSeed)?),
        _ => return Ok(false),
    }
    Ok(true)
}

pub(crate) fn response_format(value: &str) -> Option<SpeechResponseFormat> {
    match value.trim().to_ascii_lowercase().as_str() {
        "mp3" => Some(SpeechResponseFormat::Mp3),
        "opus" => Some(SpeechResponseFormat::Opus),
        "aac" => Some(SpeechResponseFormat::Aac),
        "flac" => Some(SpeechResponseFormat::Flac),
        "wav" => Some(SpeechResponseFormat::Wav),
        "pcm" => Some(SpeechResponseFormat::Pcm),
        _ => None,
    }
}

pub(crate) fn task(value: &str) -> Option<SpeechTask> {
    let normalized = value.trim().replace(['_', '-'], "").to_ascii_lowercase();
    match normalized.as_str() {
        "base" => Some(SpeechTask::TextToSpeech),
        "customvoice" => Some(SpeechTask::VoiceClone),
        "voicedesign" => Some(SpeechTask::VoiceDesign),
        _ => None,
    }
}

pub(crate) fn reference_forms(fields: &SpeechFields) -> Vec<ReferenceForm> {
    collect_reference_forms(
        fields.ref_audio.as_ref().is_some_and(Option::is_some),
        fields.references.flatten(),
    )
}

pub(crate) fn effective_reference_forms(
    defaults: &SpeechFields,
    item: &SpeechFields,
) -> Vec<ReferenceForm> {
    let has_ref_audio = item
        .ref_audio
        .as_ref()
        .and_then(Option::as_ref)
        .or_else(|| defaults.ref_audio.as_ref().and_then(Option::as_ref))
        .is_some();
    let references = item
        .references
        .flatten()
        .or_else(|| defaults.references.flatten());
    collect_reference_forms(has_ref_audio, references)
}

fn collect_reference_forms(
    has_ref_audio: bool,
    references: Option<ReferenceFlags>,
) -> Vec<ReferenceForm> {
    let mut result = Vec::with_capacity(3);
    if has_ref_audio {
        result.push(ReferenceForm::Direct);
    }
    if let Some(flags) = references {
        if flags.list {
            result.push(ReferenceForm::List);
        }
        if flags.vq_codes {
            result.push(ReferenceForm::VqCodes);
        }
    }
    if result.is_empty() {
        result.push(ReferenceForm::None);
    }
    result
}

pub(crate) fn managed_voice(fields: &SpeechFields, references: &[ReferenceForm]) -> bool {
    references == [ReferenceForm::None]
        && fields
            .voice
            .as_ref()
            .and_then(Option::as_deref)
            .is_some_and(|voice| !voice.is_empty() && !voice.eq_ignore_ascii_case("default"))
}

struct NullableReferencesSeed;

impl<'de> DeserializeSeed<'de> for NullableReferencesSeed {
    type Value = Option<ReferenceFlags>;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct OptionalVisitor;
        impl<'de> Visitor<'de> for OptionalVisitor {
            type Value = Option<ReferenceFlags>;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("null or a reference array")
            }

            fn visit_none<E>(self) -> Result<Self::Value, E> {
                Ok(None)
            }

            fn visit_unit<E>(self) -> Result<Self::Value, E> {
                Ok(None)
            }

            fn visit_some<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                deserializer.deserialize_seq(ReferencesVisitor).map(Some)
            }
        }
        deserializer.deserialize_option(OptionalVisitor)
    }
}

struct ReferencesVisitor;

impl<'de> Visitor<'de> for ReferencesVisitor {
    type Value = ReferenceFlags;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a reference array")
    }

    fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut flags = ReferenceFlags::default();
        while let Some(entry) = sequence.next_element_seed(ReferenceSeed)? {
            flags.list = true;
            flags.vq_codes |= entry.vq_codes;
        }
        Ok(flags)
    }
}

struct ReferenceSeed;

impl<'de> DeserializeSeed<'de> for ReferenceSeed {
    type Value = ReferenceFlags;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct ReferenceVisitor;
        impl<'de> Visitor<'de> for ReferenceVisitor {
            type Value = ReferenceFlags;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a speech reference object")
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut flags = ReferenceFlags::default();
                while let Some(key) = map.next_key::<String>()? {
                    if matches!(key.as_str(), "audio_path" | "ref_audio" | "audio" | "data") {
                        flags.list |= map.next_value::<Option<IgnoredAny>>()?.is_some();
                    } else if key == "vq_codes" {
                        flags.vq_codes = map.next_value::<Option<IgnoredAny>>()?.is_some();
                    } else {
                        let _ignored = map.next_value::<IgnoredAny>()?;
                    }
                }
                Ok(flags)
            }
        }
        deserializer.deserialize_map(ReferenceVisitor)
    }
}
