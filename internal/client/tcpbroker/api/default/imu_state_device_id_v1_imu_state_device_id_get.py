from http import HTTPStatus
from typing import Any, Dict, Optional, Union, cast

import httpx

from ...client import Client
from ...models.http_validation_error import HTTPValidationError
from ...types import Response


def _get_kwargs(
    device_id: str,
    *,
    client: Client,
) -> Dict[str, Any]:
    url = "{}/v1/imu/state/{device_id}".format(client.base_url, device_id=device_id)

    headers: Dict[str, str] = client.get_headers()
    cookies: Dict[str, Any] = client.get_cookies()

    return {
        "method": "get",
        "url": url,
        "headers": headers,
        "cookies": cookies,
        "timeout": client.get_timeout(),
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
    device_id: str,
    *,
    client: Client,
) -> Response[Union[Any, HTTPValidationError]]:
    """Imu State Device Id

    Args:
        device_id (str):

    Returns:
        Response[Union[Any, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        device_id=device_id,
        client=client,
    )

    response = httpx.request(
        verify=client.verify_ssl,
        **kwargs,
    )

    return _build_response(response=response)


def sync(
    device_id: str,
    *,
    client: Client,
) -> Optional[Union[Any, HTTPValidationError]]:
    """Imu State Device Id

    Args:
        device_id (str):

    Returns:
        Response[Union[Any, HTTPValidationError]]
    """

    return sync_detailed(
        device_id=device_id,
        client=client,
    ).parsed


async def asyncio_detailed(
    device_id: str,
    *,
    client: Client,
) -> Response[Union[Any, HTTPValidationError]]:
    """Imu State Device Id

    Args:
        device_id (str):

    Returns:
        Response[Union[Any, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        device_id=device_id,
        client=client,
    )

    async with httpx.AsyncClient(verify=client.verify_ssl) as _client:
        response = await _client.request(**kwargs)

    return _build_response(response=response)


async def asyncio(
    device_id: str,
    *,
    client: Client,
) -> Optional[Union[Any, HTTPValidationError]]:
    """Imu State Device Id

    Args:
        device_id (str):

    Returns:
        Response[Union[Any, HTTPValidationError]]
    """

    return (
        await asyncio_detailed(
            device_id=device_id,
            client=client,
        )
    ).parsed
