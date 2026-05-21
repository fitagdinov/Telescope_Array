import asyncio
import sqlite3
from datetime import datetime
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import Message, CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.utils.keyboard import InlineKeyboardBuilder

# ========== КОНФИГУРАЦИЯ ==========
TOKEN = "8300278653:AAHKNVqIVvVHVsdLykaY2nbxaqpRHtU1yXw"
bot = Bot(token=TOKEN)
dp = Dispatcher()

# ========== БАЗА ДАННЫХ ==========
conn = sqlite3.connect("points.db")
cursor = conn.cursor()
cursor.execute("""
    CREATE TABLE IF NOT EXISTS user_points (
        chat_id INTEGER,
        user_id INTEGER,
        points INTEGER DEFAULT 0,
        PRIMARY KEY (chat_id, user_id)
    )
""")
conn.commit()

def add_point(chat_id: int, user_id: int):
    cursor.execute("""
        INSERT INTO user_points (chat_id, user_id, points)
        VALUES (?, ?, 1)
        ON CONFLICT(chat_id, user_id) DO UPDATE SET points = points + 1
    """, (chat_id, user_id))
    conn.commit()

def get_points(chat_id: int, user_id: int) -> int:
    cursor.execute("SELECT points FROM user_points WHERE chat_id=? AND user_id=?", (chat_id, user_id))
    row = cursor.fetchone()
    return row[0] if row else 0

def get_leaderboard(chat_id: int, limit=10):
    cursor.execute("""
        SELECT user_id, points FROM user_points
        WHERE chat_id=?
        ORDER BY points DESC
        LIMIT ?
    """, (chat_id, limit))
    return cursor.fetchall()

# ========== ХРАНИЛИЩЕ АКТИВНЫХ ГОЛОСОВАНИЙ ==========
active_votes = {}  # key: chat_id, value: dict

class VoteSession:
    def __init__(self, target_user_id, required_yes, message_id, total_members):
        self.target_user_id = target_user_id
        self.required_yes = required_yes
        self.message_id = message_id
        self.yes_votes = set()
        self.no_votes = set()
        self.total_members = total_members
        self.end_time = datetime.now().timestamp() + 30  # 30 секунд на голосование

# ========== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ==========
async def get_human_members_count(chat_id: int) -> int:
    """Возвращает количество людей в чате (исключая ботов)."""
    total = await bot.get_chat_member_count(chat_id)
    # Вычитаем самого бота (можно также исключить других ботов, но для простоты так)
    return total - 1

async def end_vote(chat_id: int, vote: VoteSession, bot_instance: Bot):
    """Завершает голосование, начисляет Мойш-балл при успехе."""
    yes_count = len(vote.yes_votes)
    required = vote.required_yes
    target_id = vote.target_user_id

    if yes_count >= required:
        add_point(chat_id, target_id)
        result_text = f"✅ Голосование завершено! Участнику @{target_id} начислен Мойш-балл.\n"
        result_text += f"За: {yes_count} / {required} (нужно ≥{required})"
    else:
        result_text = f"❌ Голосование не прошло. Не хватило голосов «За».\n"
        result_text += f"За: {yes_count}, нужно: {required}"

    # Обновляем сообщение с результатом
    try:
        await bot_instance.edit_message_text(
            result_text,
            chat_id=chat_id,
            message_id=vote.message_id,
            reply_markup=None
        )
    except:
        pass

    # Удаляем сессию
    if chat_id in active_votes:
        del active_votes[chat_id]

# ========== КОМАНДЫ ==========
@dp.message(Command("start"))
async def cmd_start(message: Message):
    await message.answer("👋 Я бот для подсчёта Мойш-баллов в чате.\n"
                         "Команды:\n"
                         "/points [@username] – показать Мойш-баллы\n"
                         "/vote @username – начать голосование за начисление Мойш-балла")

@dp.message(Command("points"))
async def cmd_points(message: Message):
    chat_id = message.chat.id
    args = message.text.split()
    if len(args) > 1 and args[1].startswith("@"):
        # Поиск упомянутого пользователя
        username = args[1][1:]
        try:
            member = await bot.get_chat_member(chat_id, username)
            user_id = member.user.id
            points = get_points(chat_id, user_id)
            await message.answer(f"🎯 Мойш-баллы пользователя @{username}: {points}")
        except:
            await message.answer("❌ Не удалось найти такого участника.")
    else:
        # Показать топ-10
        leaderboard = get_leaderboard(chat_id)
        if not leaderboard:
            await message.answer("Пока нет ни одного Мойш-балла у участников.")
            return
        text = "🏆 *Таблица лидеров:*\n"
        for idx, (uid, pts) in enumerate(leaderboard, 1):
            try:
                user = await bot.get_chat_member(chat_id, uid)
                name = user.user.first_name
                text += f"{idx}. {name} – {pts} Мойш-баллов\n"
            except:
                text += f"{idx}. Пользователь {uid} – {pts}\n"
        await message.answer(text, parse_mode="Markdown")

@dp.message(Command("vote"))
async def cmd_vote(message: Message):
    chat_id = message.chat.id
    args = message.text.split()
    if len(args) < 2 or not args[1].startswith("@"):
        await message.answer("❌ Укажите участника: `/vote @username`", parse_mode="Markdown")
        return

    # Проверяем, нет ли уже активного голосования в этом чате
    if chat_id in active_votes:
        await message.answer("⚠️ В этом чате уже идёт голосование. Дождитесь его завершения.")
        return

    target_username = args[1][1:]
    try:
        member = await bot.get_chat_member(chat_id, target_username)
        target_user_id = member.user.id
        target_name = member.user.first_name
    except:
        await message.answer("❌ Пользователь не найден или бот не может его увидеть.")
        return

    # Если голосуют за самого себя – разрешим, но можно запретить (раскомментировать)
    # if target_user_id == message.from_user.id:
    #     await message.answer("Нельзя голосовать за самого себя.")
    #     return

    # Получаем количество людей в чате
    try:
        human_count = await get_human_members_count(chat_id)
    except Exception as e:
        await message.answer("❌ Не удалось определить количество участников. Убедитесь, что бот администратор.")
        return

    if human_count <= 1:
        await message.answer("В чате мало людей для голосования (нужно хотя бы 2 участника).")
        return

    # Половина участников, округление в меньшую сторону: 4→2, 5→2, 6→3
    required_yes = human_count // 2

    # Создаём клавиатуру
    builder = InlineKeyboardBuilder()
    builder.add(InlineKeyboardButton(text="✅ За", callback_data="vote_yes"))
    builder.add(InlineKeyboardButton(text="❌ Против", callback_data="vote_no"))
    builder.adjust(2)

    vote_msg = await message.answer(
        f"🗳️ Голосование: начислить Мойш-балл {target_name} (@{target_username})?\n"
        f"Требуется голосов «За»: {required_yes} (половина от {human_count} участников, округление вниз)\n"
        f"Время: 3 часа ",
        reply_markup=builder.as_markup()
    )

    # Сохраняем сессию
    vote_session = VoteSession(
        target_user_id=target_user_id,
        required_yes=required_yes,
        message_id=vote_msg.message_id,
        total_members=human_count
    )
    active_votes[chat_id] = vote_session

    # Запускаем таймер завершения
    asyncio.create_task(delayed_end_vote(chat_id, vote_session))

async def delayed_end_vote(chat_id: int, vote_session: VoteSession):
    await asyncio.sleep(60*60*3)
    if chat_id in active_votes and active_votes[chat_id] == vote_session:
        await end_vote(chat_id, vote_session, bot)

# ========== ОБРАБОТЧИК КНОПОК ГОЛОСОВАНИЯ ==========
@dp.callback_query(lambda c: c.data in ["vote_yes", "vote_no"])
async def process_vote_button(callback: CallbackQuery):
    chat_id = callback.message.chat.id
    user_id = callback.from_user.id
    vote_data = active_votes.get(chat_id)

    if not vote_data:
        await callback.answer("Голосование уже завершено или неактивно.", show_alert=False)
        await callback.message.edit_reply_markup(reply_markup=None)
        return

    # Проверяем, не голосовал ли уже
    if user_id in vote_data.yes_votes or user_id in vote_data.no_votes:
        await callback.answer("Вы уже проголосовали в этом голосовании!", show_alert=True)
        return

    # Записываем голос
    if callback.data == "vote_yes":
        vote_data.yes_votes.add(user_id)
        await callback.answer("Вы проголосовали ЗА")
    else:
        vote_data.no_votes.add(user_id)
        await callback.answer("Вы проголосовали ПРОТИВ")

    # Обновляем сообщение с текущей статистикой
    yes_cnt = len(vote_data.yes_votes)
    no_cnt = len(vote_data.no_votes)
    total_voted = yes_cnt + no_cnt
    remaining = vote_data.total_members - total_voted

    text = callback.message.text.split("\n")[0] + "\n"  # сохраняем заголовок
    text += f"✅ За: {yes_cnt}   ❌ Против: {no_cnt}\n"
    text += f"Осталось голосов: {remaining}\n"
    text += f"Нужно За: {vote_data.required_yes}\n(голосование закончится через таймер)"

    await callback.message.edit_text(text, reply_markup=callback.message.reply_markup)

    # Если набралось достаточно "За" – можно завершить досрочно (опционально)
    if yes_cnt >= vote_data.required_yes:
        await end_vote(chat_id, vote_data, bot)

# ========== ЗАПУСК ==========
async def main():
    print("Бот запущен")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())