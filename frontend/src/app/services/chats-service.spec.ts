import { TestBed } from '@angular/core/testing';
import { HttpClientTestingModule, HttpTestingController } from '@angular/common/http/testing';
import { ChatsService } from './chats-service';
import { ChatItem } from '../models/chat-item';
import { ChatMessageItem } from '../models/chat-message-item';

describe('Chats', () => {
  let service: ChatsService;
  let httpMock: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      imports: [HttpClientTestingModule],
      providers: [ChatsService]
    });
    service = TestBed.inject(ChatsService);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    httpMock.verify();
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });

  it('should list chats', async () => {
    const mockChats: ChatItem[] = [{ id: '1', title: 'Chat 1' } as ChatItem];
    const promise = service.listChats();

    const req = httpMock.expectOne('/api/chat/list');
    expect(req.request.method).toBe('GET');
    req.flush({ chats: mockChats });

    const result = await promise;
    expect(result).toEqual(mockChats);
  });

  it('should list messages', async () => {
    const mockMessages: ChatMessageItem[] = [{ role: 'user', content: 'hello' } as ChatMessageItem];
    const chatId = '123';
    const promise = service.listMessages(chatId);

    const req = httpMock.expectOne(`/api/chat/messages/${chatId}`);
    req.flush({ chat_id: chatId, chat_messages: mockMessages });

    const result = await promise;
    expect(result).toEqual(mockMessages);
  });

  it('should run agent', async () => {
    const mockInput = { key: 'value' };
    const mockResponse = { result: 'ok' };
    const promise = service.runAgent(mockInput);

    const req = httpMock.expectOne('/api/agent/run');
    expect(req.request.method).toBe('POST');
    expect(req.request.body).toEqual(mockInput);
    req.flush(mockResponse);

    const result = await promise;
    expect(result).toEqual(mockResponse);
  });
});
