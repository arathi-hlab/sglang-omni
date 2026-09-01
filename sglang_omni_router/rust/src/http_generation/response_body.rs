use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Duration;

use bytes::Bytes;
use http_body::{Frame, SizeHint};
use thiserror::Error;

use crate::worker_pool::RequestLease;

use super::request_body::{SharedUploadState, UploadState};

#[derive(Debug, Error)]
#[error("upstream response body terminated")]
pub(crate) struct RelayError;

/// Direct upstream response body whose terminal owner retains admission.
pub(crate) struct DirectResponseBody {
    inner: Option<reqwest::Body>,
    lease: Option<RequestLease>,
    upload: Option<SharedUploadState>,
    upload_deadline: Option<Pin<Box<tokio::time::Sleep>>>,
    response_idle_timeout: Option<Duration>,
    response_idle_deadline: Option<Pin<Box<tokio::time::Sleep>>>,
    terminal: bool,
}

impl DirectResponseBody {
    pub(crate) fn new(
        inner: reqwest::Body,
        lease: RequestLease,
        upload: Option<SharedUploadState>,
        deadline: tokio::time::Instant,
        response_idle_timeout: Option<Duration>,
    ) -> Self {
        let upload_deadline = upload
            .as_ref()
            .map(|_| Box::pin(tokio::time::sleep_until(deadline)));
        let response_idle_deadline = if upload.is_none() {
            response_idle_timeout.map(tokio::time::sleep).map(Box::pin)
        } else {
            None
        };
        Self {
            inner: Some(inner),
            lease: Some(lease),
            upload,
            upload_deadline,
            response_idle_timeout,
            response_idle_deadline,
            terminal: false,
        }
    }

    fn terminalize(&mut self, upstream_failure: bool) {
        if self.terminal {
            return;
        }
        self.terminal = true;
        if upstream_failure && let Some(lease) = self.lease.as_ref() {
            lease.request_immediate_probe();
        }
        drop(self.inner.take());
        drop(self.lease.take());
        drop(self.upload.take());
        drop(self.upload_deadline.take());
        drop(self.response_idle_deadline.take());
    }

    fn fail(&mut self, upstream_failure: bool) -> Poll<Option<Result<Frame<Bytes>, RelayError>>> {
        self.terminalize(upstream_failure);
        Poll::Ready(Some(Err(RelayError)))
    }

    fn poll_upload(&mut self, cx: &mut Context<'_>) -> Result<(), RelayError> {
        let Some(upload) = self.upload.as_ref() else {
            return Ok(());
        };
        let state = upload.poll_state(cx).map_err(|_| RelayError)?;
        match state {
            UploadState::Complete => {
                self.upload = None;
                self.upload_deadline = None;
                self.reset_response_idle();
                Ok(())
            }
            UploadState::Failed(_) => Err(RelayError),
            UploadState::Incomplete => {
                let expired = self
                    .upload_deadline
                    .as_mut()
                    .is_none_or(|deadline| deadline.as_mut().poll(cx).is_ready());
                if expired { Err(RelayError) } else { Ok(()) }
            }
        }
    }

    fn reset_response_idle(&mut self) {
        self.response_idle_deadline = self
            .response_idle_timeout
            .map(tokio::time::sleep)
            .map(Box::pin);
    }

    fn response_idle_expired(&mut self, cx: &mut Context<'_>) -> bool {
        self.response_idle_deadline
            .as_mut()
            .is_some_and(|deadline| deadline.as_mut().poll(cx).is_ready())
    }
}

impl Drop for DirectResponseBody {
    fn drop(&mut self) {
        self.terminalize(false);
    }
}

impl http_body::Body for DirectResponseBody {
    type Data = Bytes;
    type Error = RelayError;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        if self.terminal {
            return Poll::Ready(None);
        }
        if self.poll_upload(cx).is_err() {
            return self.fail(false);
        }
        let Some(inner) = self.inner.as_mut() else {
            return self.fail(true);
        };
        match Pin::new(inner).poll_frame(cx) {
            Poll::Ready(Some(Ok(frame))) => match frame.into_data() {
                Ok(data) => {
                    if self.upload.is_none() {
                        self.reset_response_idle();
                    }
                    Poll::Ready(Some(Ok(Frame::data(data))))
                }
                Err(_trailers) => self.fail(true),
            },
            Poll::Ready(Some(Err(_source))) => self.fail(true),
            Poll::Ready(None) => {
                self.terminalize(false);
                Poll::Ready(None)
            }
            Poll::Pending if self.response_idle_expired(cx) => self.fail(true),
            Poll::Pending => Poll::Pending,
        }
    }

    fn is_end_stream(&self) -> bool {
        self.terminal
    }

    fn size_hint(&self) -> SizeHint {
        SizeHint::default()
    }
}
