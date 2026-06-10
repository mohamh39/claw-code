//! Medtronic CLP-odin API Provider
//! Translates claw-code requests to odin/Bedrock format with Azure AD auth.

use std::sync::{Arc, RwLock};
use serde::{Deserialize, Serialize};
use crate::error::ApiError;
use crate::http_client::build_http_client_or_default;
use crate::types::{
    InputContentBlock, InputMessage, MessageRequest, MessageResponse, OutputContentBlock, Usage,
    StreamEvent, MessageStartEvent, MessageDeltaEvent, MessageStopEvent,
    ContentBlockStartEvent, ContentBlockDeltaEvent, ContentBlockStopEvent,
    ContentBlockDelta, MessageDelta, ToolDefinition, ToolChoice,
};
use super::{dotenv_value, Provider, ProviderFuture};
use std::collections::VecDeque;

// Default endpoint - configurable via ODIN_BASE_URL in .env
pub const DEFAULT_ODIN_URL: &str = "https://vpce-0a74b154c6fd02d2d-k72zrw5i.execute-api.us-east-1.vpce.amazonaws.com/stageus";
pub const ODIN_GATEWAY_ID: &str = "89f15wmsb7";

#[derive(Debug)]
struct TokenInfo { token: String, expiry: u64 }

#[derive(Debug)]
pub struct OdinProvider {
    http_client: reqwest::Client,
    endpoint: String,
    gateway_id: String,
    azure_client_id: String,
    azure_secret: String,
    azure_tenant: String,
    azure_scope: String,
    cached_token: Arc<RwLock<Option<TokenInfo>>>,
}

#[derive(Serialize)]
struct OdinReqBody {
    model_name: String,
    messages: Vec<OdinMsg>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<Vec<SysContent>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    client_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<ToolDefinition>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<ToolChoice>,
}

#[derive(Serialize, Deserialize)]
struct OdinMsg { role: String, content: Vec<MsgContent> }

#[derive(Serialize, Deserialize)]
struct MsgContent { text: String }

#[derive(Serialize)]
struct SysContent { text: String }

#[derive(Deserialize)]
struct OdinResp {
    content: Vec<OdinContent>,
    #[serde(default)]
    usage: Option<UsageInfo>,
    #[serde(default, rename = "stopReason")]
    stop_reason: Option<String>,
}

#[derive(Deserialize)]
struct OdinContent {
    #[serde(default)]
    text: Option<String>,
}

#[derive(Deserialize)]
struct UsageInfo {
    #[serde(default)]
    input_tokens: Option<u32>,
    #[serde(default)]
    output_tokens: Option<u32>,
    #[serde(default)]
    total_tokens: Option<u32>,
}

#[derive(Deserialize)]
struct AzureTokenResp { access_token: String, expires_in: u64 }

fn current_epoch() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn env_or_dotenv(k: &str) -> Option<String> {
    std::env::var(k).ok().filter(|v| !v.is_empty()).or_else(|| dotenv_value(k))
}

impl OdinProvider {
    pub fn try_from_env() -> Result<Self, ApiError> {
        // ODIN_ENV toggle: "prod" or "stage" (default: stage)
        let env_mode = env_or_dotenv("ODIN_ENV").unwrap_or_else(|| "stage".into());
        let is_prod = env_mode.to_lowercase() == "prod";
        
        // Load credentials based on environment toggle
        let (cid, sec, ten, scp, url, gw) = if is_prod {
            (
                env_or_dotenv("ODIN_CLIENT_ID")
                    .or_else(|| env_or_dotenv("ODIN_PROD_CLIENT_ID"))
                    .ok_or_else(|| ApiError::missing_credentials("odin", &["ODIN_PROD_CLIENT_ID"]))?,
                env_or_dotenv("ODIN_CLIENT_SECRET")
                    .or_else(|| env_or_dotenv("ODIN_PROD_CLIENT_SECRET"))
                    .ok_or_else(|| ApiError::missing_credentials("odin", &["ODIN_PROD_CLIENT_SECRET"]))?,
                env_or_dotenv("ODIN_TENANT_ID")
                    .or_else(|| env_or_dotenv("ODIN_PROD_TENANT_ID"))
                    .ok_or_else(|| ApiError::missing_credentials("odin", &["ODIN_PROD_TENANT_ID"]))?,
                env_or_dotenv("ODIN_SCOPE")
                    .or_else(|| env_or_dotenv("ODIN_PROD_SCOPE"))
                    .ok_or_else(|| ApiError::missing_credentials("odin", &["ODIN_PROD_SCOPE"]))?,
                env_or_dotenv("ODIN_BASE_URL")
                    .or_else(|| env_or_dotenv("ODIN_PROD_BASE_URL"))
                    .unwrap_or_else(|| DEFAULT_ODIN_URL.into()),
                env_or_dotenv("ODIN_GATEWAY_ID")
                    .or_else(|| env_or_dotenv("ODIN_PROD_GATEWAY_ID"))
                    .unwrap_or_else(|| ODIN_GATEWAY_ID.into()),
            )
        } else {
            (
                env_or_dotenv("ODIN_CLIENT_ID")
                    .or_else(|| env_or_dotenv("ODIN_STAGE_CLIENT_ID"))
                    .or_else(|| env_or_dotenv("ODIN_NONPROD_CLIENT_ID"))
                    .ok_or_else(|| ApiError::missing_credentials("odin", &["ODIN_STAGE_CLIENT_ID"]))?,
                env_or_dotenv("ODIN_CLIENT_SECRET")
                    .or_else(|| env_or_dotenv("ODIN_STAGE_CLIENT_SECRET"))
                    .or_else(|| env_or_dotenv("ODIN_NONPROD_CLIENT_SECRET"))
                    .ok_or_else(|| ApiError::missing_credentials("odin", &["ODIN_STAGE_CLIENT_SECRET"]))?,
                env_or_dotenv("ODIN_TENANT_ID")
                    .or_else(|| env_or_dotenv("ODIN_STAGE_TENANT_ID"))
                    .or_else(|| env_or_dotenv("ODIN_NONPROD_TENANT_ID"))
                    .ok_or_else(|| ApiError::missing_credentials("odin", &["ODIN_STAGE_TENANT_ID"]))?,
                env_or_dotenv("ODIN_SCOPE")
                    .or_else(|| env_or_dotenv("ODIN_STAGE_SCOPE"))
                    .or_else(|| env_or_dotenv("ODIN_NONPROD_SCOPE"))
                    .unwrap_or_else(|| "api://4d5801ef-ff38-4754-9671-67d944e7e66e/.default".into()),
                env_or_dotenv("ODIN_BASE_URL")
                    .or_else(|| env_or_dotenv("ODIN_STAGE_BASE_URL"))
                    .unwrap_or_else(|| DEFAULT_ODIN_URL.into()),
                env_or_dotenv("ODIN_GATEWAY_ID")
                    .or_else(|| env_or_dotenv("ODIN_STAGE_GATEWAY_ID"))
                    .unwrap_or_else(|| ODIN_GATEWAY_ID.into()),
            )
        };

        Ok(Self {
            http_client: build_http_client_or_default(),
            endpoint: url,
            gateway_id: gw,
            azure_client_id: cid,
            azure_secret: sec,
            azure_tenant: ten,
            azure_scope: scp,
            cached_token: Arc::new(RwLock::new(None)),
        })
    }

    pub fn credentials_available() -> bool {
        env_or_dotenv("ODIN_CLIENT_ID").is_some() || env_or_dotenv("ODIN_NONPROD_CLIENT_ID").is_some()
    }

    async fn acquire_azure_token(&self) -> Result<String, ApiError> {
        // Return cached if valid
        if let Ok(guard) = self.cached_token.read() {
            if let Some(ref info) = *guard {
                if info.expiry > current_epoch() + 300 {
                    return Ok(info.token.clone());
                }
            }
        }
        // Request fresh token
        let token_endpoint = format!(
            "https://login.microsoftonline.com/{}/oauth2/v2.0/token",
            self.azure_tenant
        );
        let form_data = [
            ("client_id", self.azure_client_id.as_str()),
            ("client_secret", self.azure_secret.as_str()),
            ("scope", self.azure_scope.as_str()),
            ("grant_type", "client_credentials"),
        ];
        let resp = self.http_client.post(&token_endpoint).form(&form_data).send().await
            .map_err(ApiError::Http)?;
        if !resp.status().is_success() {
            let st = resp.status();
            let bd = resp.text().await.unwrap_or_default();
            return Err(ApiError::Auth(format!("Azure auth failed: {st} {bd}")));
        }
        let parsed: AzureTokenResp = resp.json().await
            .map_err(ApiError::Http)?;
        // Cache it
        if let Ok(mut guard) = self.cached_token.write() {
            *guard = Some(TokenInfo {
                token: parsed.access_token.clone(),
                expiry: current_epoch() + parsed.expires_in,
            });
        }
        Ok(parsed.access_token)
    }

    fn resolve_odin_model(&self, alias: &str) -> String {
        let lc = alias.to_ascii_lowercase();
        
        // Check for env-configured model aliases first (adaptive)
        let sonnet = env_or_dotenv("ODIN_MODEL_SONNET")
            .unwrap_or_else(|| "us.anthropic.claude-sonnet-4-6".into());
        let opus = env_or_dotenv("ODIN_MODEL_OPUS")
            .unwrap_or_else(|| "us.anthropic.claude-opus-4-6-v1".into());
        let haiku = env_or_dotenv("ODIN_MODEL_HAIKU")
            .unwrap_or_else(|| "us.anthropic.claude-haiku-4-5-20251001-v1:0".into());
        let default = env_or_dotenv("ODIN_DEFAULT_MODEL")
            .unwrap_or_else(|| sonnet.clone());

        match lc.as_str() {
            // odin aliases - use env-configured values
            "odin" | "odin-sonnet" | "odin/sonnet" | "sonnet" | "claude-sonnet" => sonnet,
            "odin-opus" | "odin/opus" | "opus" | "claude-opus" => opus,
            "odin-haiku" | "odin/haiku" | "haiku" | "claude-haiku" => haiku,
            // Already odin format - pass through
            s if s.starts_with("us.anthropic.") || s.starts_with("anthropic.") => alias.to_string(),
            // Claude 3.5 (fallback defaults)
            s if s.contains("3-5-sonnet") => "us.anthropic.claude-3-5-sonnet-20241022-v2:0".into(),
            s if s.contains("3-5-haiku") => "us.anthropic.claude-3-5-haiku-20241022-v1:0".into(),
            // Claude 3 (fallback defaults)
            s if s.contains("3-opus") => "anthropic.claude-3-opus-20240229-v1:0".into(),
            s if s.contains("3-sonnet") => "anthropic.claude-3-sonnet-20240229-v1:0".into(),
            s if s.contains("3-haiku") => "anthropic.claude-3-haiku-20240307-v1:0".into(),
            // Default
            _ => default,
        }
    }

    fn convert_messages(&self, msgs: &[InputMessage]) -> Vec<OdinMsg> {
        msgs.iter().map(|m| {
            let c: Vec<MsgContent> = m.content.iter().filter_map(|b| {
                if let InputContentBlock::Text { text, .. } = b { Some(MsgContent { text: text.clone() }) } else { None }
            }).collect();
            OdinMsg { role: m.role.clone(), content: c }
        }).collect()
    }

    fn convert_system(&self, sys: &Option<String>) -> Option<Vec<SysContent>> {
        sys.as_ref().map(|text| vec![SysContent { text: text.clone() }])
    }

    async fn invoke(&self, req: &MessageRequest) -> Result<MessageResponse, ApiError> {
        let bearer = self.acquire_azure_token().await?;
        let api_url = format!("{}/inferences", self.endpoint.trim_end_matches('/'));
        let body = OdinReqBody {
            model_name: self.resolve_odin_model(&req.model),
            messages: self.convert_messages(&req.messages),
            system: self.convert_system(&req.system),
            temperature: req.temperature.map(|t| t as f32),
            max_tokens: Some(req.max_tokens),
            client_id: self.azure_client_id.clone(),
            tools: req.tools.clone(),
            tool_choice: req.tool_choice.clone(),
        };
        let http_resp = self.http_client.post(&api_url)
            .bearer_auth(&bearer)
            .header("Content-Type", "application/json")
            .header("x-apigw-api-id", &self.gateway_id)
            .json(&body)
            .send().await
            .map_err(ApiError::Http)?;
        if !http_resp.status().is_success() {
            let st = http_resp.status();
            let bd = http_resp.text().await.unwrap_or_default();
            return Err(ApiError::Api {
                status: st,
                error_type: Some("odin_error".to_string()),
                message: Some(bd.clone()),
                request_id: None,
                body: bd,
                retryable: st.is_server_error(),
                suggested_action: None,
                retry_after: None,
            });
        }
        let odin_resp: OdinResp = http_resp.json().await
            .map_err(ApiError::Http)?;
        let blocks: Vec<OutputContentBlock> = odin_resp.content.iter()
            .filter_map(|c| c.text.as_ref().map(|t| OutputContentBlock::Text { text: t.clone() }))
            .collect();
        let usg = odin_resp.usage.map(|u| Usage {
            input_tokens: u.input_tokens.unwrap_or(0),
            output_tokens: u.output_tokens.unwrap_or(0),
            cache_creation_input_tokens: 0,
            cache_read_input_tokens: 0,
        }).unwrap_or_default();
        Ok(MessageResponse {
            id: format!("odin-{}", current_epoch()),
            kind: "message".to_string(),
            role: "assistant".to_string(),
            content: blocks,
            model: req.model.clone(),
            stop_reason: odin_resp.stop_reason,
            stop_sequence: None,
            usage: usg,
            request_id: None,
        })
    }
}

impl Provider for OdinProvider {
    type Stream = OdinMessageStream;

    fn send_message<'a>(&'a self, request: &'a MessageRequest) -> ProviderFuture<'a, MessageResponse> {
        Box::pin(self.invoke(request))
    }

    fn stream_message<'a>(&'a self, request: &'a MessageRequest) -> ProviderFuture<'a, Self::Stream> {
        Box::pin(async move {
            // Call the API and get full response
            let response = self.invoke(request).await?;
            Ok(OdinMessageStream::from_response(response))
        })
    }
}

#[derive(Debug)]
pub struct OdinMessageStream {
    events: VecDeque<StreamEvent>,
    done: bool,
}

impl OdinMessageStream {
    /// Create a stream from a complete MessageResponse, simulating streaming events
    fn from_response(response: MessageResponse) -> Self {
        let mut events = VecDeque::new();
        
        // MessageStart event
        events.push_back(StreamEvent::MessageStart(MessageStartEvent {
            message: MessageResponse {
                id: response.id.clone(),
                kind: response.kind.clone(),
                role: response.role.clone(),
                content: vec![],
                model: response.model.clone(),
                stop_reason: None,
                stop_sequence: None,
                usage: Usage::default(),
                request_id: None,
            },
        }));
        
        // Emit content blocks
        for (index, block) in response.content.iter().enumerate() {
            if let OutputContentBlock::Text { text } = block {
                let idx = index as u32;
                // ContentBlockStart
                events.push_back(StreamEvent::ContentBlockStart(ContentBlockStartEvent {
                    index: idx,
                    content_block: OutputContentBlock::Text { text: String::new() },
                }));
                
                // ContentBlockDelta with full text
                events.push_back(StreamEvent::ContentBlockDelta(ContentBlockDeltaEvent {
                    index: idx,
                    delta: ContentBlockDelta::TextDelta { text: text.clone() },
                }));
                
                // ContentBlockStop
                events.push_back(StreamEvent::ContentBlockStop(ContentBlockStopEvent { index: idx }));
            }
        }
        
        // MessageDelta with usage and stop reason
        events.push_back(StreamEvent::MessageDelta(MessageDeltaEvent {
            delta: MessageDelta {
                stop_reason: response.stop_reason.clone(),
                stop_sequence: response.stop_sequence.clone(),
            },
            usage: response.usage.clone(),
        }));
        
        // MessageStop
        events.push_back(StreamEvent::MessageStop(MessageStopEvent {}));
        
        Self { events, done: false }
    }
    
    pub fn request_id(&self) -> Option<&str> {
        None
    }
    
    pub async fn next_event(&mut self) -> Result<Option<StreamEvent>, ApiError> {
        if self.done {
            return Ok(None);
        }
        
        if let Some(event) = self.events.pop_front() {
            if matches!(event, StreamEvent::MessageStop(_)) {
                self.done = true;
            }
            Ok(Some(event))
        } else {
            self.done = true;
            Ok(None)
        }
    }
}
