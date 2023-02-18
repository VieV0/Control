from aiogram import Bot, Dispatcher, executor, types
import logging
from constants import *
import control

bot = Bot(token=TOKEN, parse_mode=types.ParseMode.HTML)
dp = Dispatcher(bot)
logging.basicConfig(level=logging.INFO)

@dp.message_handler(commands="start")
async def start(message: types.Message):
    await message.answer(f"Здравствуйте, {message.from_user.full_name}!\nДанный бот предназначен для прогнозирования эффективности сотрудника.\nДля использования бота напишите:\n<strong>[Опыт (количество лет)]</strong>, <strong>[Концентрируемость (по 10-ти бальной шкале)]</strong>, <strong>[Сфера деятельности (для информации напишите команду <u>/foa</u>)]</strong>.")

@dp.message_handler(commands="foa")
async def start(message: types.Message):
    await message.answer(FOA)

@dp.message_handler()
async def mmain(message: types.Message):
    data = []
    if ", " in message.text:
        data = message.text.split(", ")
    elif "," in message.text:
        data = message.text.split(",")
    elif " " in message.text:
        data = message.text.split(" ")
    else:
        await message.answer("<strong>Введены некорректные данные!</strong>\nПожалуйста, повторите попытку.")
        return

    data_number = control.np.array([float(data[0]), float(data[1]), float(data[2])])
    data_number /= 10.0

    result = control.sess.run(control.y_, feed_dict = {control.X: control.np.array([data_number]).transpose()})
    await message.answer(f"Предполагаемая эффективность: {result[0][0]*100}%.")

if __name__ == "__main__":
    executor.start_polling(dp, skip_updates=True)