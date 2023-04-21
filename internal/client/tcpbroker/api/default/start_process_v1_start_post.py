from http import HTTPStatus
from typing import Any, Dict, Optional, Union, cast

import httpx

from ...client import Client
from ...models.http_validation_error import HTTPValidationError
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    client: Client,
    tag: Union[Unset, None, str] = UNSET,
    experiment_log: Union[Unset, None, str] = UNSET,
) -> Dict[str, Any]:
    url = "{}/v1/start".format(client.base_url)

    headers: Dict[str, str] = client.get_headers()
    cookies: Dict[str, Any] = client.get_cookies()

    params: Dict[str, Any] = {}
    params["tag"] = tag

    params["experiment_log"] = experiment_log

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    return {
        "method": "post",
        "url": url,
        "headers": headers,
        "cookies": cookies,
        "timeout": client.get_timeout(),
        "params": params,
    }


def _parse_response(*, response: httpx.Response) -> Optional[Union[Any, HTTPValidationError]]:
    if response.status_code == HTTPStatus.OK:
        response_200 = cast(Any, response.json())
        return response_200
    if response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY:
        response_422 = HTTPValidationError.from_dict(response.json())

        return response_422
    return None


def _build_response(*, response: httpx.Response) -> Response[Union[Any, HTTPValidationError]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(response=response),
    )


def sync_detailed(
    *,
    client: Client,
    tag: Union[Unset, None, str] = UNSET,
    experiment_log: Union[Unset, None, str] = UNSET,
) -> Response[Union[Any, HTTPValidationError]]:
    """Start Process

    Args:
        tag (Union[Unset, None, str]):
        experiment_log (Union[Unset, None, str]):

    Returns:
        Response[Union[Any, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        client=client,
        tag=tag,
        experiment_log=experiment_log,
    )

    response = httpx.request(
        verify=client.verify_ssl,
        **kwargs,
    )

    return _build_response(response=response)


def sync(
    *,
    client: Client,
    tag: Union[Unset, None, str] = UNSET,
    experiment_log: Union[Unset, None, str] = UNSET,
) -> Optional[Union[Any, HTTPValidationError]]:
    """Start Process

    Args:
        tag (Union[Unset, None, str]):
        experiment_log (Union[Unset, None, str]):

    Returns:
        Response[Union[Any, HTTPValidationError]]
    """

    return sync_detailed(
        client=client,
        tag=tag,
        experiment_log=experiment_log,
    ).parsed


async def asyncio_detailed(
    *,
    client: Client,
    tag: Union[Unset, None, str] = UNSET,
    experiment_log: Union[Unset, None, str] = UNSET,
) -> Response[Union[Any, HTTPValidationError]]:
    """Start Process

    Args:
        tag (Union[Unset, None, str]):
        experiment_log (Union[Unset, None, str]):

    Returns:
        Response[Union[Any, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        client=client,
        tag=tag,
        experiment_log=experiment_log,
    )

    async with httpx.AsyncClient(verify=client.verify_ssl) as _client:
        response = await _client.request(**kwargs)

    return _build_response(response=response)


async def asyncio(
    *,
    client: Client,
    tag: Union[Unset, None, str] = UNSET,
    experiment_log: Union[Unset, None, str] = UNSET,
) -> Optional[Union[Any, HTTPValidationError]]:
    """Start Process

    Args:
        tag (Union[Unset, None, str]):
        experiment_log (Union[Unset, None, str]):

    Returns:
        Response[Union[Any, HTTPValidationError]]
    """

    return (
        await asyncio_detailed(
            client=client,
            tag=tag,
            experiment_log=experiment_log,
        )
    ).parsed