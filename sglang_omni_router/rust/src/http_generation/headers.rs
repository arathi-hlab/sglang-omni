use axum::http::header::{
    CONTENT_ENCODING, CONTENT_LENGTH, CONTENT_TYPE, EXPECT, TRAILER, TRANSFER_ENCODING,
};
use axum::http::{HeaderMap, HeaderValue, StatusCode};

use crate::error::HttpFault;
use crate::http_relay::{is_request_media_type, parse_content_length, sanitize_response_headers};

pub(crate) struct RequestFraming {
    pub(crate) content_length: Option<u64>,
}

pub(crate) fn validate_request(headers: &HeaderMap) -> Result<RequestFraming, HttpFault> {
    let mut content_types = headers.get_all(CONTENT_TYPE).iter();
    let content_type = content_types.next();
    if content_types.next().is_some()
        || !content_type
            .and_then(|value| value.to_str().ok())
            .is_some_and(|value| is_request_media_type(value, "application/json"))
    {
        return Err(HttpFault::UnsupportedMediaType);
    }
    let mut encodings = headers.get_all(CONTENT_ENCODING).iter();
    let encoding = encodings.next();
    if encodings.next().is_some()
        || encoding.is_some_and(|value| {
            !value
                .to_str()
                .is_ok_and(|text| text.eq_ignore_ascii_case("identity"))
        })
    {
        return Err(HttpFault::UnsupportedContentEncoding);
    }
    let mut expectations = headers.get_all(EXPECT).iter();
    if let Some(expectation) = expectations.next()
        && (!expectation.as_bytes().eq_ignore_ascii_case(b"100-continue")
            || expectations.next().is_some())
    {
        return Err(HttpFault::ExpectationFailed);
    }
    if headers.contains_key(TRAILER) {
        return Err(HttpFault::MalformedRequest);
    }
    let transfer_framed = headers.contains_key(TRANSFER_ENCODING);
    let mut content_lengths = headers.get_all(CONTENT_LENGTH).iter();
    let content_length = content_lengths.next();
    if content_lengths.next().is_some() || (transfer_framed && content_length.is_some()) {
        return Err(HttpFault::MalformedRequest);
    }
    let content_length = content_length
        .map(|value| parse_content_length(value).ok_or(HttpFault::MalformedRequest))
        .transpose()?;
    if content_length.is_none() && !transfer_framed {
        return Err(HttpFault::MalformedRequest);
    }
    Ok(RequestFraming { content_length })
}

pub(crate) fn sanitize_response(
    status: StatusCode,
    source: &HeaderMap,
) -> Result<HeaderMap, HttpFault> {
    sanitize_response_headers(status, source)
}

pub(crate) fn canonical_content_type() -> HeaderValue {
    HeaderValue::from_static("application/json")
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use axum::http::header::{
        CACHE_CONTROL, CONTENT_ENCODING, CONTENT_LENGTH, CONTENT_TYPE, EXPECT, TRANSFER_ENCODING,
    };
    use axum::http::{HeaderMap, HeaderValue, StatusCode};

    use super::{HttpFault, sanitize_response, validate_request};

    fn valid_request_headers() -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(
            CONTENT_TYPE,
            HeaderValue::from_static("application/json; charset=UTF-8"),
        );
        headers.insert(CONTENT_LENGTH, HeaderValue::from_static("12"));
        headers
    }

    #[test]
    fn request_envelope_accepts_fixed_and_chunked_identity_encoded_json() {
        let headers = valid_request_headers();
        assert_eq!(
            validate_request(&headers)
                .expect("valid fixed request")
                .content_length,
            Some(12)
        );

        for (name, value, fault) in [
            (CONTENT_TYPE, "text/plain", HttpFault::UnsupportedMediaType),
            (
                CONTENT_ENCODING,
                "gzip",
                HttpFault::UnsupportedContentEncoding,
            ),
        ] {
            let mut rejected = headers.clone();
            rejected.insert(name, HeaderValue::from_static(value));
            assert_eq!(validate_request(&rejected).err(), Some(fault));
        }

        let mut identity = headers.clone();
        identity.insert(CONTENT_ENCODING, HeaderValue::from_static("identity"));
        assert!(validate_request(&identity).is_ok());

        let mut chunked = headers.clone();
        chunked.remove(CONTENT_LENGTH);
        chunked.insert(TRANSFER_ENCODING, HeaderValue::from_static("chunked"));
        assert_eq!(
            validate_request(&chunked)
                .expect("chunked request")
                .content_length,
            None
        );

        let mut conflicting = headers.clone();
        conflicting.insert(TRANSFER_ENCODING, HeaderValue::from_static("chunked"));
        assert_eq!(
            validate_request(&conflicting).err(),
            Some(HttpFault::MalformedRequest)
        );

        let mut missing_length = headers;
        missing_length.remove(CONTENT_LENGTH);
        assert_eq!(
            validate_request(&missing_length).err(),
            Some(HttpFault::MalformedRequest)
        );

        let mut duplicate_type = valid_request_headers();
        duplicate_type.append(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        assert_eq!(
            validate_request(&duplicate_type).err(),
            Some(HttpFault::UnsupportedMediaType)
        );
        let mut duplicate_length = valid_request_headers();
        duplicate_length.append(CONTENT_LENGTH, HeaderValue::from_static("12"));
        assert_eq!(
            validate_request(&duplicate_length).err(),
            Some(HttpFault::MalformedRequest)
        );
    }

    #[test]
    fn request_envelope_accepts_only_one_standard_expectation() {
        let mut accepted = valid_request_headers();
        accepted.insert(EXPECT, HeaderValue::from_static("100-Continue"));
        assert!(validate_request(&accepted).is_ok());

        for value in ["continue", "100-continue, custom"] {
            let mut rejected = valid_request_headers();
            rejected.insert(EXPECT, HeaderValue::from_static(value));
            assert_eq!(
                validate_request(&rejected).err(),
                Some(HttpFault::ExpectationFailed)
            );
        }

        let mut duplicate = valid_request_headers();
        duplicate.append(EXPECT, HeaderValue::from_static("100-continue"));
        duplicate.append(EXPECT, HeaderValue::from_static("100-continue"));
        assert_eq!(
            validate_request(&duplicate).err(),
            Some(HttpFault::ExpectationFailed)
        );
    }

    #[test]
    fn gateway_hints_are_ignored_at_the_client_boundary() {
        let mut headers = valid_request_headers();
        headers.insert(
            "x-smg-target-worker",
            HeaderValue::from_static("external-choice"),
        );
        headers.insert("x-smg-routing-key", HeaderValue::from_static("key"));
        headers.insert(
            "x-sgl-decode-url",
            HeaderValue::from_static("http://untrusted.invalid"),
        );
        assert!(validate_request(&headers).is_ok());
    }

    #[test]
    fn response_sanitizer_preserves_end_to_end_headers_and_safe_duplicates() {
        let mut source = HeaderMap::new();
        source.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        source.append(CACHE_CONTROL, HeaderValue::from_static("private"));
        source.append(CACHE_CONTROL, HeaderValue::from_static("max-age=0"));
        source.insert(CONTENT_ENCODING, HeaderValue::from_static("gzip"));
        source.insert("set-cookie", HeaderValue::from_static("private=1"));
        source.insert("connection", HeaderValue::from_static("x-private"));
        source.insert("x-private", HeaderValue::from_static("secret"));

        let sanitized = sanitize_response(StatusCode::OK, &source).expect("valid worker response");
        assert_eq!(sanitized.get_all(CACHE_CONTROL).iter().count(), 2);
        assert_eq!(
            sanitized.get(CONTENT_ENCODING),
            source.get(CONTENT_ENCODING)
        );
        assert_eq!(sanitized.get("set-cookie"), source.get("set-cookie"));
        assert!(!sanitized.contains_key("x-private"));
    }

    #[test]
    fn response_preserves_worker_media_type_and_rejects_invalid_framing() {
        let mut plain = HeaderMap::new();
        plain.insert(CONTENT_TYPE, HeaderValue::from_static("text/plain"));
        let sanitized = sanitize_response(StatusCode::OK, &plain)
            .expect("worker-owned response type is relayable");
        assert_eq!(sanitized.get(CONTENT_TYPE), plain.get(CONTENT_TYPE));
        assert_eq!(
            sanitize_response(StatusCode::TEMPORARY_REDIRECT, &HeaderMap::new()).err(),
            Some(HttpFault::UpstreamProtocolError)
        );

        let mut duplicate_type = HeaderMap::new();
        duplicate_type.append(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        duplicate_type.append(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        assert_eq!(
            sanitize_response(StatusCode::OK, &duplicate_type).err(),
            Some(HttpFault::UpstreamProtocolError)
        );
        let mut duplicate_length = HeaderMap::new();
        duplicate_length.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        duplicate_length.append(CONTENT_LENGTH, HeaderValue::from_static("2"));
        duplicate_length.append(CONTENT_LENGTH, HeaderValue::from_static("2"));
        assert_eq!(
            sanitize_response(StatusCode::OK, &duplicate_length).err(),
            Some(HttpFault::UpstreamProtocolError)
        );
    }

    #[test]
    fn worker_errors_relay_without_a_success_content_type() {
        let headers = HeaderMap::new();
        let sanitized = sanitize_response(StatusCode::UNPROCESSABLE_ENTITY, &headers)
            .expect("worker error response is relayable");
        assert!(sanitized.is_empty());
    }

    #[test]
    fn successful_json_and_sse_responses_are_relayable() {
        let mut json = HeaderMap::new();
        json.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        let mut sse = HeaderMap::new();
        sse.insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
        assert!(sanitize_response(StatusCode::OK, &json).is_ok());
        assert!(sanitize_response(StatusCode::OK, &sse).is_ok());
    }
}
